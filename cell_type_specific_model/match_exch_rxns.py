from cobra.io import read_sbml_model, write_sbml_model
import pandas as pd
import logging
from copy import deepcopy


import os
import sys
from cobra import Reaction

# Determine the current file's directory and the project root.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
# Add the project root to sys.path to access top-level folders like 'functions' and 'models'
if project_root not in sys.path:
    sys.path.append(project_root)

from functions.function_metabolite_identification import generate_met_annotation, process_annotation
from functions.functions_merge_metabolic_networks import network_metabolites_merge_3, network_genes_merge_2, network_reactions_merge_7
reports_dir = os.path.join(project_root, 'metabolite_reac_identification', 'reports')

def match_exch_rxns(model_new, model_base, tol=1e-9, add_rs: bool = False):
    """
    Match and update exchange reaction bounds by metabolite enrichment and network merging,
    using original feasibility checks and bound-relax logic for matched reactions.

    Parameters
    ----------
    model_new : cobra.Model
        The model to modify.
    model_base : cobra.Model
        The reference model whose bounds will be applied.
    tol : float
        Tolerance for objective feasibility checks.
    add_rs : bool
        If True, add boundary reactions to the model_new that are in the model_base.
    Returns
    -------
    model_modified : cobra.Model
        The updated model_new with new exchange reaction bounds.
    eq_rxns : list of tuples
        Pairs of (new_rxn_id, base_rxn_id) that were matched.
    noneq_rxns : list of str
        Exchange reaction IDs in model_new that had no match.
    inconsistent_rxns : list of str
        Reactions flagged as inconsistent during merging.
    """
    # Copy model for modifications
    model_modified = model_new.copy()
    logging.info(f"Total boundary reactions: {len(model_new.boundary)}")

    # 1. Identify exchange reactions (single-metabolite)
    exchange_rxns = [rxn for rxn in model_new.boundary if len(rxn.metabolites) == 1]
    logging.info(f"Identified {len(exchange_rxns)} exchange reactions")
    # 2. Enrich metabolites annotations
    met_list = [
        (met.name, met.formula, met.annotation, met.id)
        for rxn in exchange_rxns for met in rxn.metabolites
    ]

    
    annotated, unannotated = generate_met_annotation(met_list, out=os.path.join(reports_dir, 'met_annotation.tsv'))
    logging.info(f"Annotated {len(annotated)} metabolites, unannotated: {len(unannotated)}")

    cwd = os.getcwd()                                    # save current dir
    os.chdir(project_root)
    try:
        annotation_map = process_annotation()            # now it finds reports/…
    finally:
        os.chdir(cwd)                                    # restore original dir
    logging.info(f"Annotation map number of metabolites: {len(annotation_map)}")

    for met_id, ann in annotation_map.items():
        if met_id in model_modified.metabolites:
            model_modified.metabolites.get_by_id(met_id).annotation.update(ann)

    # 3. Build equivalence maps
    _, eq_meta = network_metabolites_merge_3(model_modified.copy(), model_base.copy())
    logging.info(f"Matched {len(eq_meta)} metabolites")


    def core_id_from_model(met_id: str, model) -> str:
        """
        Return the compartment–free core ID for `met_id` by consulting the model.
        Works for bracket, underscore, and flat-suffix styles.
        """
        try:
            comp = model.metabolites.get_by_id(met_id).compartment
        except KeyError:
            return met_id                    # unknown → leave unchanged

        if met_id.endswith(f'_{comp}'):
            return met_id[:-(len(comp) + 1)]
        if met_id.endswith(f'[{comp}]'):
            return met_id[:-(len(comp) + 2)]
        if met_id.endswith(comp):
            return met_id[:-len(comp)]
        return met_id


    # 1. build map:  base-core → new-ID (underscore tags kept)
    meta_to_new = {}
    for new_id, base_id in eq_meta:                 # (new, base) from merge function
        base_core = core_id_from_model(base_id, model_base)
        new_core  = core_id_from_model(new_id,  model_new)
        meta_to_new[base_core] = new_core


    # 2. count boundary metabolites that are in the map
    seen_base, seen_new = set(), set()
    for r in model_base.boundary:
        if len(r.metabolites) == 1:
            core = core_id_from_model(next(iter(r.metabolites)).id, model_base)
            if core in meta_to_new:
                seen_base.add(core)

    for r in model_new.boundary:
        if len(r.metabolites) == 1:
            core = core_id_from_model(next(iter(r.metabolites)).id, model_new)
            if core in meta_to_new.values():
                seen_new.add(core)

    logging.info(f"Matched {len(seen_base)} metabolites in base-model boundary reactions")
    logging.info(f"Matched {len(seen_new)} metabolites in new-model boundary reactions")

    # 3. Find equivalent exchange reactions by metabolite mapping
    eq_ex_rxns = []
    ex_new  = [r for r in model_new.boundary  if len(r.metabolites) == 1]
    ex_base = [r for r in model_base.boundary if len(r.metabolites) == 1]

    for r_b in ex_base:
        # base metabolite + compartment
        met_b   = next(iter(r_b.metabolites))
        core_b  = core_id_from_model(met_b.id, model_base)
        comp_b  = met_b.compartment

        # must have a mapping
        if core_b not in meta_to_new:
            continue

        target_core = meta_to_new[core_b]

        # look for a new‐model reaction on the same core **and** same compartment
        for r_n in ex_new:
            met_n  = next(iter(r_n.metabolites))
            core_n = core_id_from_model(met_n.id, model_new)
            comp_n = met_n.compartment

            # both core and compartment must match
            if core_n == target_core and comp_n == comp_b:
                eq_ex_rxns.append((r_b.id, r_n.id))
                break

    logging.info(f"Matched {len(eq_ex_rxns)} boundary reactions via metabolites")

    def rtype(r):
        """Return 'EX', 'DM', 'SK', or None for single-metabolite boundary reactions."""
        if len(r.metabolites) != 1:
            return None
        if r.id.startswith('EX_'):
            return 'EX'
        if r.id.startswith('DM_'):
            return 'DM'
        if r.id.startswith('SK_'):
            return 'SK'
        # fallback: decide from metabolite compartment
        comp = next(iter(r.metabolites)).compartment.lower()
        return 'EX' if comp in {'e', 'x', 'p', 'ext'} else 'DM'


    # 4. Apply bounds matching with try/except and relaxation logic
    reactions_found = 0
    reactions_failed = []
    for base_id,new_id in eq_ex_rxns:
        try:
            # ensure reactions exist
            rxn_new = model_modified.reactions.get_by_id(new_id)
            rxn_base = model_base.reactions.get_by_id(base_id)
             #  only copy bounds when the reaction categories match
            if rtype(rxn_base) != rtype(rxn_new):
                logging.info(f"Skip: {base_id} ({rtype(rxn_base)}) → {new_id} ({rtype(rxn_new)})")
                continue
            # save original bounds
            original_bounds = deepcopy(rxn_new.bounds)
            base_bounds = deepcopy(rxn_base.bounds)
            # set to base-model bounds
            rxn_new.lower_bound, rxn_new.upper_bound = base_bounds

            # test feasibility
            opt = model_modified.optimize()
            if opt.objective_value is None or opt.objective_value < tol:
                logging.warning(f'Reaction not feasible: {new_id}')
                # relax bounds incrementally
                increment = 0.1
                start_lb, start_ub = rxn_new.bounds
                while opt.objective_value is None or opt.objective_value < tol:
                    # compute new relaxed bounds
                    lb = round(start_lb - increment * abs(base_bounds[0]), 7)
                    ub = round(start_ub + increment * abs(base_bounds[1]), 7)
                    rxn_new.lower_bound = 0 if abs(lb) < tol else lb
                    rxn_new.upper_bound = 0 if abs(ub) < tol else ub
                    opt = model_modified.optimize()
                    if increment >= 10:
                        bounds_not_found = True
                        break
                    increment += 0.1
                else:
                    bounds_not_found = False
                    logging.warning(f'Reaction feasible after increment: {new_id}')
                    logging.info(f'Base bounds: {base_bounds}, New bounds: {rxn_new.bounds}, Obj: {opt.objective_value}')

                if bounds_not_found:
                    logging.warning(f'Reaction not feasible after increment: {new_id}')
                    # revert to original
                    rxn_new.lower_bound, rxn_new.upper_bound = original_bounds
                    opt2 = model_modified.optimize()
                    if opt2.objective_value is None or opt2.objective_value < tol:
                        logging.error(f'Reaction still infeasible after revert: {new_id}')
                        reactions_failed.append(new_id)
                else:
                    reactions_found += 1
            else:
                reactions_found += 1
        except Exception as e:
            logging.error(f'Error processing reaction {new_id}: {e}')
            reactions_failed.append(new_id)

    logging.info(f'After attempts for bounds matching: Out of the {len(eq_ex_rxns)} reactions processed:')
    logging.info(f'Matched reactions: {reactions_found}, Failed: {len(reactions_failed)}')


    # 5. Add missing boundary reactions from base model to new model


    if add_rs:

        # figure out the next MAR counter
        # grab all existing MAR IDs in the model, e.g. “MAR09079”
        mar_ids = [
            rxn.id for rxn in model_modified.reactions
            if rxn.id.startswith("MAR") and rxn.id[3:].isdigit()
        ]
        # extract their numeric parts
        nums = [int(mid[3:]) for mid in mar_ids]
        max_num = max(nums) if nums else 0
        # width is how many digits the  MAR codes use (e.g. 5)
        width = len(mar_ids[0]) - 3 if mar_ids else 5
        next_counter = max_num + 1

        # Find which base‐cores never got a match  
        matched_cores = {
            core_id_from_model(
                next(iter(model_base.reactions.get_by_id(b).metabolites)).id,
                model_base
            )
            for b, _ in eq_ex_rxns
        }

        unmatched_cores = seen_base - matched_cores
        print(f"{len(unmatched_cores)} cores still missing a boundary reaction match")

        # For each missing core, clone its base boundary rxn into model_modified  
        for core in unmatched_cores:
            # find all base boundary reactions carrying that core
            for r_b in ex_base:
                met_b = next(iter(r_b.metabolites))
                if core_id_from_model(met_b.id, model_base) != core:
                    continue

                # lookup the new‐model core
                new_core = meta_to_new.get(core)
                if new_core is None:
                    # no mapping even at the metabolite level
                    continue

                # find the full new‐model metabolite ID with same compartment
                candidates = [
                    m for m in model_modified.metabolites
                    if core_id_from_model(m.id, model_new) == new_core
                    and m.compartment == met_b.compartment
                ]
                if not candidates:
                    continue
                new_met_id = candidates[0].id
                met_obj    = model_modified.metabolites.get_by_id(new_met_id)

                # make sure we don’t duplicate a same‐type rxn
                t = rtype(r_b)
                exists = any(
                    len(r.metabolites)==1
                    and rtype(r)==t
                    and core_id_from_model(
                        next(iter(r.metabolites)).id, model_new
                    ) == new_core
                    for r in model_modified.boundary
                )
                if exists:
                    continue

                # use the MAR‐code of the metabolite as the reaction ID 
                new_rxn_id = f"MAR{next_counter:0{width}d}"
                # ensure it’s unique
                while new_rxn_id in model_modified.reactions:
                    next_counter += 1
                    new_rxn_id = f"MAR{next_counter:0{width}d}"

                rxn_new = Reaction(new_rxn_id)
                rxn_new.name        = f"Imported from {r_b.id}"
                rxn_new.lower_bound, rxn_new.upper_bound = r_b.bounds
                coeff = r_b.metabolites[met_b]
                rxn_new.add_metabolites({met_obj: coeff})

                model_modified.add_reactions([rxn_new])
                logging.info(f"Added missing {t} reaction {new_rxn_id} for core {core}")

                # register it so your bounds‐copy loop will include it
                eq_ex_rxns.append((r_b.id, new_rxn_id))

    logging.info(f"Modified {len(eq_ex_rxns)} reactions in new model")
    #noneq_rxns are the reactions in the exchange rxns that are not in the eq_rxns list
    exchange_rxns_ids = [rxn.id for rxn in exchange_rxns]
    noneq_rxns = set(exchange_rxns_ids) - set([rxn[1] for rxn in eq_ex_rxns])
    logging.info(f'Non-equivalent reactions: {len(noneq_rxns)}')
    

    # Zero-out unmatched reactions with revert logic
    for rxn_id in noneq_rxns:
        try:
            rxn = model_modified.reactions.get_by_id(rxn_id)
            orig = deepcopy(rxn.bounds)
            rxn.lower_bound, rxn.upper_bound = 0, 0
            opt0 = model_modified.optimize()
            if opt0.objective_value is None or opt0.objective_value < tol:
                logging.warning(f'Model infeasible setting 0 bounds: {rxn_id}')
                # revert
                rxn.lower_bound, rxn.upper_bound = orig
                opt1 = model_modified.optimize()
                if opt1.objective_value is None or opt1.objective_value < tol:
                    logging.error(f'Infeasible after revert: {rxn_id}, bounds: {orig}')
        except Exception as e:
            logging.error(f'Error zeroing reaction {rxn_id}: {e}')


    # 6. Prepare original metabolite DataFrame and common list

    metab_df = get_metab_df(model_modified, noneq_rxns)
    common_rs_list = [new_id for _, new_id in eq_ex_rxns]
    common_rs_list = list(set(common_rs_list))  # remove duplicates
    eq_ex_rxns = list(set(eq_ex_rxns))  # remove duplicates
    logging.info(f'Common reactions number: {len(common_rs_list)}')
    logging.info(f'Non-equivalent reactions number: {len(set(noneq_rxns))}')

    # Return both original and new outputs
    return model_modified, metab_df, common_rs_list, eq_ex_rxns, noneq_rxns



def get_metab_df(model, reactions):
    '''
    Get a dataframe of metabolites and the reactions they are involved in.

    Parameters
    ----------
    model : cobra.Model
        The model to use.
    reactions : list
        List of reaction ids.
    '''
    # Calculate the total number of metabolites
    total_metabolites = sum(len(model.reactions.get_by_id(reaction_id).metabolites) for reaction_id in reactions)
    
    # Preassign list lengths
    metab_ids = [None] * total_metabolites
    metab_names = [None] * total_metabolites
    metab_kegg = [None] * total_metabolites
    metab_seed = [None] * total_metabolites
    metab_bigg = [None] * total_metabolites
    metab_chebi = [None] * total_metabolites
    metab_pubchem = [None] * total_metabolites
    metab_vmh = [None] * total_metabolites
    metab_inchi = [None] * total_metabolites
    metab_inchikey = [None] * total_metabolites

    reaction_ids = [None] * total_metabolites
    reaction_names = [None] * total_metabolites
    reaction_ubs = [None] * total_metabolites
    reaction_lbs = [None] * total_metabolites
    reaction_bigg = [None] * total_metabolites


    index = 0
    for reaction_id in reactions:
        reaction = model.reactions.get_by_id(reaction_id)
        metab = list(reaction.metabolites.keys())[0]
        for metab in reaction.metabolites:
            metab_ids[index] = metab.id
            metab_names[index] = metab.name
            metab_kegg[index] = metab.annotation.get('kegg.compound')
            metab_seed[index] = metab.annotation.get('seed.compound')
            metab_bigg[index] = metab.annotation.get('bigg.metabolite')
            metab_chebi[index] = metab.annotation.get('chebi')
            metab_pubchem[index] = metab.annotation.get('pubchem.compound')
            metab_vmh[index] = metab.annotation.get('vmh.metabolite')
            metab_inchi[index] = metab.annotation.get('inchi')
            metab_inchikey[index] = metab.annotation.get('inchikey')

            reaction_ids[index] = reaction_id
            reaction_names[index] = reaction.name
            reaction_ubs[index] = reaction.upper_bound
            reaction_lbs[index] = reaction.lower_bound
            reaction_bigg[index] = reaction.annotation.get('bigg.reaction')
            index += 1

    metab_df = pd.DataFrame({
        'Metabolite IDs': metab_ids, 
        'Metabolite names': metab_names, 
        'Metabolite KeggIDs': metab_kegg, 
        'Metabolite SeedIDs': metab_seed, 
        'Metabolite BiGGIDs': metab_bigg, 
        'Metabolite ChebiIDs': metab_chebi,
        'Metabolite PubchemIDs': metab_pubchem,
        'Metabolite VMHIDs': metab_vmh,
        'Metabolite Inchi': metab_inchi,
        'Metabolite Inchikey': metab_inchikey,

        'Reaction IDs': reaction_ids,
        'Reaction names': reaction_names, 
        'Reaction UBs': reaction_ubs, 
        'Reaction LBs': reaction_lbs
    })
    return metab_df

if __name__ == '__main__':


    model_base = read_sbml_model(os.path.join(project_root, 'models','EC_model_with_KEGG.xml'))
    model_new = read_sbml_model(os.path.join(project_root, 'models','THG-beta2_endoA.xml'))  
   
    #use Gurobi as solver
    model_new.solver = 'gurobi'
    opt = model_new.optimize()
    logging.basicConfig(level=logging.DEBUG)
    print('Initial objective value:', opt.objective_value)
    return_model, metab_df, common_rs, eq_ex_rxns, noneq_rxns = match_exch_rxns(model_new, model_base, tol=1e-9, add_rs=True)  
    
    sol = return_model.optimize()
    print('Final new model objective value:', sol.objective_value)
    sol_base = model_base.optimize()
    print('Base model objective value:', sol_base.objective_value)
    write_sbml_model(return_model, os.path.join(project_root, 'models','THG_endoA_boundary.xml'))
    #metab_df.to_csv('pipeline/data/EC_THG_not_found_metabolites.csv', index=False)
    metab_df.to_excel('EC_THG_not_found_metabolites.xlsx', index=False)

    #write common reactions to file for debug
    with open(os.path.join(project_root, 'files','common_rs.txt'), 'w') as f:
        for item in common_rs:
            f.write("%s\n" % item)

    print('Done')



