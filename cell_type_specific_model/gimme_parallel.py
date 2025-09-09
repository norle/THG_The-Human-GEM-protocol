import os
import numpy as np
import pandas as pd
import cobra
from concurrent.futures import ProcessPoolExecutor
from cobra.util import create_stoichiometric_matrix
import pdb
from cobra.flux_analysis import flux_variability_analysis
import multiprocessing
import sys
import copy
from troppo.methods.reconstruction.gimme import GIMME, GIMMEProperties

# Determine the current file's directory and the project root.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
# Add the project root to sys.path to access top-level folders like 'functions' and 'models'
if project_root not in sys.path:
    sys.path.append(project_root)


def build_expression_dict_from_original(original_rxn_ids, expression_rxns_sample):
    """
    Build a dictionary mapping each original reaction ID to its corresponding
    expression value from a given 1D expression vector (for one sample).

    Parameters:
        original_rxn_ids (list): A list of reaction IDs in their original order.
        expression_rxns_sample (iterable): A 1D array-like object containing the
                                             expression values corresponding to each
                                             reaction in original_rxn_ids.

    Returns:
        dict: A dictionary with reaction IDs as keys and their expression values as values.
    """
    return {
        rxn_id: expression_rxns_sample[i] for i, rxn_id in enumerate(original_rxn_ids)
    }


def get_extended_scores(modified_model, original_expr_dict):
    """
    Generate an extended expression vector for a modified model (which may include split reactions).
    For reactions that were split (i.e. IDs ending in "_fwd" or "_rev"), the expression value is taken
    from the base reaction in the original expression dictionary.

    Parameters:
        modified_model (cobra.Model): The modified metabolic model.
        original_expr_dict (dict): A dictionary mapping original reaction IDs to expression values.

    Returns:
        np.array: A numpy array of expression values for all reactions in the modified model.
    """
    extended_scores = []
    for rxn in modified_model.reactions:
        if rxn.id in original_expr_dict:
            extended_scores.append(float(original_expr_dict[rxn.id]))
        else:
            if rxn.id.endswith("_fwd") or rxn.id.endswith("_rev"):
                base_id = rxn.id.rsplit("_", 1)[0]
                extended_scores.append(float(original_expr_dict.get(base_id, 0.0)))
            else:
                extended_scores.append(0.0)
    return np.array(extended_scores)


def split_reversible_exchanges(
    model,
    original_expr_dict,
    exchange_reactions,
    min_bound=1e-5,
    fva_filename=os.path.join(project_root, "files", "reversible_fva.csv"),
):
    """
    Splits reversible exchange reactions into two irreversible reactions.
    Uses batch FVA (or loads precomputed results from fva_filename) to determine the flux capacity
    in each direction.

    For each reversible reaction in exchange_reactions:
      - Uses FVA to obtain the maximum and minimum fluxes.
      - For the forward reaction: if the FVA maximum > 0, set its lower bound to min_bound; otherwise, 0.
      - For the reverse reaction: if the FVA minimum < 0, set its lower bound to min_bound and its upper bound
        to the absolute value of the FVA minimum; otherwise, 0.

    After creating forward and reverse copies, the original reaction is removed and the new ones are added.

    Parameters:
        model (cobra.Model): The metabolic model to modify.
        original_expr_dict (dict): Expression values for the original reactions.
        exchange_reactions (list): List of exchange reaction IDs to process.
        min_bound (float): Minimum flux bound to enforce (default 1e-5).
        fva_filename (str): Filename for saving/loading FVA results.

    Returns:
        tuple: (modified model, split_mapping, updated_expression)
            - modified model: The updated cobra.Model after splitting.
            - split_mapping (dict): Mapping of original reaction IDs to a tuple (forward_rxn_id, reverse_rxn_id).
            - updated_expression (np.array): Updated expression vector for the modified model.
    """
    # Filter out the reversible exchange reactions.
    reversible_exchanges = [
        rxn
        for rxn in exchange_reactions
        if model.reactions.get_by_id(rxn).reversibility
    ]
    split_mapping = {}

    # Load precomputed FVA results if available; otherwise, compute and save them.
    if os.path.exists(fva_filename):
        fva_results = pd.read_csv(fva_filename, index_col=0)
    else:
        # Ensure the model is in a clean state before running FVA.
        model.solver.problem.reset()
        fva_results = flux_variability_analysis(
            model, reaction_list=reversible_exchanges, processes=1
        )
        fva_results.to_csv(fva_filename)

    for rxn in reversible_exchanges:
        rxn = model.reactions.get_by_id(rxn)

        # Retrieve FVA results for this reaction.
        fva_max = fva_results.loc[rxn.id, "maximum"]
        fva_min = fva_results.loc[rxn.id, "minimum"]

        # Set forward lower bound if the reaction is capable of carrying positive flux.
        forward_lb = float(min_bound if fva_max > 0 else 0.0)
        # Set reverse lower bound if the reaction is capable of carrying negative flux.
        reverse_lb = float(min_bound if fva_min < 0 else 0.0)
        # For the reverse reaction, set the upper bound to the absolute value of the minimum flux (if negative).
        reverse_ub = float(abs(fva_min) if fva_min < 0 else 0.0)

        # Create forward reaction copy.
        forward_rxn = rxn.copy()
        forward_rxn.id = rxn.id + "_fwd"
        forward_rxn.lower_bound = forward_lb  # Enforce a minimum forward flux.

        # Create reverse reaction copy.
        reverse_rxn = rxn.copy()
        reverse_rxn.id = rxn.id + "_rev"
        # Invert the stoichiometry.
        reverse_rxn._metabolites = {
            met: -coeff for met, coeff in rxn.metabolites.items()
        }
        reverse_rxn.lower_bound = reverse_lb
        reverse_rxn.upper_bound = reverse_ub

        # Remove original reaction and add the new ones.
        model.reactions.remove(rxn)
        model.add_reactions([forward_rxn, reverse_rxn])

        split_mapping[rxn.id] = (forward_rxn.id, reverse_rxn.id)

    updated_expression = get_extended_scores(model, original_expr_dict)
    return model, split_mapping, updated_expression


def update_irreversible_exchanges(
    model,
    exchange_reactions,
    min_bound=1e-5,
    fva_filename=os.path.join(project_root, "files", "irreversible_rxns_fva.csv"),
):
    """
    Updates the bounds of irreversible exchange reactions based on FVA analysis.

    For each irreversible exchange reaction present in the model:
      - Load FVA results from a file or compute them if not available.
      - If the reaction's FVA maximum flux is > 0 and its bounds are not consistently
        positive or negative, update its lower bound to min_bound.

    Parameters:
        model (cobra.Model): The metabolic model.
        exchange_reactions (list): List of exchange reaction IDs to check.
        min_bound (float): The minimum flux bound to assign.
        fva_filename (str): Filename for saving/loading FVA results.

    Returns:
        cobra.Model: The model with updated irreversible exchange reaction bounds.
    """

    # Filter for irreversible exchange reaction IDs from the provided list.
    irreversible_rxn_ids = [
        rxn_id
        for rxn_id in exchange_reactions
        if not model.reactions.get_by_id(rxn_id).reversibility
    ]

    # Load or compute FVA results.
    if os.path.exists(fva_filename):
        fva_results = pd.read_csv(fva_filename, index_col=0)
    else:
        fva_results = flux_variability_analysis(
            model, reaction_list=irreversible_rxn_ids
        )
        fva_results.to_csv(fva_filename)

    # Loop over each irreversible reaction and update its lower bound if needed.
    for rxn_id in irreversible_rxn_ids:
        rxn = model.reactions.get_by_id(rxn_id)

        # If both bounds are consistently positive or consistently negative, do nothing.
        if (rxn.lower_bound > 0 and rxn.upper_bound > 0) or (
            rxn.lower_bound < 0 and rxn.upper_bound < 0
        ):
            continue

        # Retrieve the FVA results for this reaction.
        if rxn.id in fva_results.index:
            min_flux = fva_results.loc[rxn.id, "minimum"]
            max_flux = fva_results.loc[rxn.id, "maximum"]
        else:
            continue  # Skip if FVA results are not available for this reaction.

        # If the FVA minimum flux is > 0, update the lower bound to min_bound.
        if max_flux > 0:
            rxn.lower_bound = min(
                min_bound, rxn.upper_bound
            )  # Ensure lower bound does not exceed upper bound.

    return model


def recombine_solution(
    solution, original_rxn_ids, split_mapping, reaction_ids_modified
):
    """
    Recombine the flux solution from the modified model (with default names "R0", "R1", …)
    into a net flux vector for the original reaction order.

    For split reactions (present in split_mapping), the net flux is:
       net_flux = flux(forward reaction) - flux(reverse reaction)
    For non-split reactions, the flux is directly taken from the solution.

    Parameters:
        solution (dict): Flux solution with keys like "R0", "R1", etc.
        original_rxn_ids (list): The original reaction IDs.
        split_mapping (dict): Mapping from original reaction IDs to (forward_rxn_id, reverse_rxn_id).
        reaction_ids_modified (list): The modified reaction names (e.g. "R0", "R1", …).

    Returns:
        np.array: Net flux values in the order of original_rxn_ids.
    """
    # Build a lookup: modified reaction id -> its index in reaction_ids_modified
    mod_index = {rxn: i for i, rxn in enumerate(reaction_ids_modified)}
    net_solution = []
    for idx, orig_rxn in enumerate(original_rxn_ids):
        if orig_rxn in split_mapping:
            fwd_rxn, rev_rxn = split_mapping[orig_rxn]
            fwd_idx = mod_index.get(fwd_rxn)
            rev_idx = mod_index.get(rev_rxn)
            fwd_flux = (
                solution.get("R" + str(fwd_idx), 0.0) if fwd_idx is not None else 0.0
            )
            rev_flux = (
                solution.get("R" + str(rev_idx), 0.0) if rev_idx is not None else 0.0
            )
            net_flux = fwd_flux - rev_flux
        else:
            # For non-split reactions, use the index in the original reaction order.
            net_flux = solution.get("R" + str(idx), 0.0)
        net_solution.append(net_flux)
    return np.array(net_solution)


# Define a worker function that will be run in parallel for each sample.
def gimme_worker(
    sample_idx,
    sample_expr,
    original_rxn_ids,
    model_copy,
    reaction_ids,
    metabolite_ids,
    split_mapping,
    S,
    lb,
    ub,
    objectives,
    obj_frac,
    flux_threshold,
):
    """
    Worker function to run the GIMME algorithm in parallel.

    Parameters:
        sample_idx (int): Index of the sample.
        sample_expr (array): Expression vector for the sample.
        original_rxn_ids (list): Original reaction IDs.
        model_copy (cobra.Model): A copy of the modified metabolic model.
        reaction_ids (list): Precomputed reaction IDs from the model.
        metabolite_ids (list): Precomputed metabolite IDs from the model.
        split_mapping (dict): Mapping from original reaction IDs to split reaction IDs.
        S (scipy.sparse matrix): Stoichiometric matrix of the model.
        lb (np.array): Array of lower bounds for reactions.
        ub (np.array): Array of upper bounds for reactions.
        objectives (list): Weighted objective function as a list of dictionaries.
        obj_frac (float): Objective fraction parameter.
        flux_threshold (float): Threshold to determine active reactions.

    Returns:
        sample_idx (int), net_solution (np.array) where net_solution is the recombined flux solution for the sample.
    """

    try:
        print(f"Running GIMME for sample {sample_idx + 1}...")

        # If a path to an SBML file was passed accidentally, load it here.
        if isinstance(model_copy, str):
            try:
                print(f"model_copy is a path string, loading SBML from {model_copy}")
                model_copy = cobra.io.read_sbml_model(model_copy)
            except Exception as e_load:
                print(f"Failed to load model from path {model_copy}: {e_load}")
                raise

        # Reset the solver to ensure clean state
        model_copy.solver.problem.reset()

        # Build the sample-specific expression dictionary.
        expr_dict = build_expression_dict_from_original(original_rxn_ids, sample_expr)

        # Compute the extended scores for the modified model.
        scores = get_extended_scores(model_copy, expr_dict)
        print(f"Number of reactions (including splits): {len(scores)}")
        print(
            f"For threshold {flux_threshold}: active reactions = {sum((scores > flux_threshold) & (scores > -1))}. Number of uncertain reactions = {sum(scores == -1)}"
        )

        # Validate reaction and metabolite IDs (precomputed and passed as arguments).
        if not isinstance(reaction_ids, list) or not all(
            isinstance(r, str) for r in reaction_ids
        ):
            raise ValueError("reaction_ids must be a list of strings.")
        if not isinstance(metabolite_ids, list) or not all(
            isinstance(m, str) for m in metabolite_ids
        ):
            raise ValueError("metabolite_ids must be a list of strings.")

        # Debug: Log the indices of the objective reactions.
        print(f"Objective indices: {objectives}")
        print(f"Stoichiometric matrix shape: {S.shape}")
        print(f"Reaction bounds: lb size = {len(lb)}, ub size = {len(ub)}")

        # Ensure objective indices are within bounds.
        for obj in objectives:
            for rxn_idx in obj.keys():
                if rxn_idx >= S.shape[1]:
                    raise IndexError(
                        f"Objective index {rxn_idx} is out of bounds for the model with {S.shape[1]} reactions."
                    )

        # Ensure `reaction_ids_modified` is defined in the scope.
        reaction_ids_modified = [rxn.id for rxn in model_copy.reactions]

        # Since objectives are already defined with correct modified model indices,
        # we just need to validate they are within bounds
        for obj in objectives:
            for rxn_idx in obj.keys():
                if rxn_idx >= S.shape[1]:
                    raise IndexError(
                        f"Objective index {rxn_idx} is out of bounds for the model with {S.shape[1]} reactions."
                    )

        # Debug: Log the objective indices.
        print(f"Final objective indices: {objectives}")

        # Set up the GIMME properties.
        properties = GIMMEProperties(
            exp_vector=scores,
            obj_frac=obj_frac,
            objectives=objectives,
            preprocess=False,
            flux_threshold=flux_threshold,
            solver="GUROBI",
            reaction_ids=reaction_ids,  # Use precomputed reaction IDs.
            metabolite_ids=metabolite_ids,  # Add metabolite IDs.
        )

        # Create and run the GIMME algorithm with robust error handling.
        gimme_instance = GIMME(S=S, lb=lb, ub=ub, properties=properties)

        # Quick pre-run sanity checks
        try:
            print("Pre-run checks:")
            print(" S.shape:", S.shape)
            print(" len(reaction_ids):", len(reaction_ids))
            print(" len(lb):", len(lb), "len(ub):", len(ub))
            print(" len(model_copy.reactions):", len([r for r in model_copy.reactions]))
            if (
                len(reaction_ids) != S.shape[1]
                or len(lb) != S.shape[1]
                or len(ub) != S.shape[1]
            ):
                print(
                    "WARNING: mismatch between stoichiometric matrix columns and reaction/bound vectors"
                )
            print("sample of reaction_ids:", reaction_ids[:10])
        except Exception as pre_check_err:
            print("Pre-run check failed:", pre_check_err)

        try:
            gimme_instance.run()
        except Exception as run_err:
            errmsg = str(run_err)
            print(f"GIMME run failed for sample {sample_idx + 1}: {errmsg}")
            print("Full traceback for initial failure:")
            import traceback

            traceback.print_exc()

            # DEBUG DUMP: Collect solver and model internals safely to help trace the error
            print("=== BEGIN DEBUG DUMP ===")
            try:
                print("Sample index:", sample_idx)
                print(
                    "Properties summary:",
                    {
                        k: getattr(properties, k)
                        for k in ["preprocess", "flux_threshold"]
                        if hasattr(properties, k)
                    },
                )
            except Exception as e_debug:
                print("Could not print properties summary:", e_debug)

            try:
                print("Type(model_copy):", type(model_copy))
                prob = getattr(model_copy.solver, "problem", None)
                print("Solver problem object:", type(prob))
                # Attempt to inspect gurobipy model if available
                if prob is not None:
                    try:
                        if hasattr(prob, "getVars"):
                            gv = prob.getVars()
                            print("gurobi var count:", len(gv))
                            print(
                                "gurobi var names (first 20):",
                                [getattr(v, "varName", str(v)) for v in gv[:20]],
                            )
                        if hasattr(prob, "getConstrs"):
                            gc = prob.getConstrs()
                            print("gurobi constr count:", len(gc))
                    except Exception as e_probe:
                        print("Failed to inspect gurobipy problem internals:", e_probe)
            except Exception as e_debug2:
                print("Failed to inspect model_copy solver object:", e_debug2)

            try:
                # Inspect gimme_instance internals
                print("gimme_instance type:", type(gimme_instance))
                attrs = [a for a in dir(gimme_instance) if not a.startswith("_")]
                print("gimme_instance public attrs sample:", attrs[:20])
                if hasattr(gimme_instance, "gm"):
                    gm = getattr(gimme_instance, "gm")
                    print("gimme_instance.gm type:", type(gm))
                    if hasattr(gm, "problem"):
                        print("gm.problem type:", type(gm.problem))
            except Exception as e_debug3:
                print("Failed to inspect gimme_instance internals:", e_debug3)

            print("=== END DEBUG DUMP ===")

            # Try to recover from Gurobi removal errors by enabling preprocessing
            if (
                "Item to be removed not a Var" in errmsg
                or "Item to be removed not a Var, Constr" in errmsg
                or "remove" in errmsg
            ):
                try:
                    print(
                        "Retrying GIMME with preprocessing enabled to avoid solver model remove issues..."
                    )
                    properties.preprocess = True
                    gimme_instance = GIMME(S=S, lb=lb, ub=ub, properties=properties)
                    gimme_instance.run()
                    print(f"Retry succeeded for sample {sample_idx + 1}")
                except Exception as run_err2:
                    print(f"Retry also failed for sample {sample_idx + 1}: {run_err2}")
                    print("Full traceback for retry failure:")
                    traceback.print_exc()
                    raise
            else:
                raise

        # Extract the solution and recombine to the original reaction order.
        try:
            # Try different ways to access the solution
            if hasattr(gimme_instance, "sol") and hasattr(
                gimme_instance.sol, "__dict__"
            ):
                if "_Solution__value_map" in gimme_instance.sol.__dict__:
                    solution = gimme_instance.sol.__dict__["_Solution__value_map"]
                elif hasattr(gimme_instance.sol, "value_map"):
                    solution = gimme_instance.sol.value_map
                elif hasattr(gimme_instance.sol, "fluxes"):
                    solution = gimme_instance.sol.fluxes
                else:
                    print("Available sol attributes:", dir(gimme_instance.sol))
                    raise AttributeError(
                        "Could not find solution values in gimme_instance.sol"
                    )
            else:
                print("gimme_instance attributes:", dir(gimme_instance))
                raise AttributeError("Could not find sol attribute in gimme_instance")

        except Exception as sol_err:
            print(f"Failed to extract solution: {sol_err}")
            # Try alternative access methods
            try:
                if hasattr(gimme_instance, "gm") and hasattr(
                    gimme_instance.gm, "solution"
                ):
                    solution = gimme_instance.gm.solution
                else:
                    raise AttributeError("No alternative solution access found")
            except Exception as alt_err:
                print(f"Alternative solution access also failed: {alt_err}")
                raise

        net_solution = recombine_solution(
            solution, original_rxn_ids, split_mapping, reaction_ids_modified
        )

        return sample_idx, net_solution

    except Exception as e:
        print(f"Error in gimme_worker for sample {sample_idx}: {e}")
        return sample_idx, None


# Main parallelized GIMME function.
def gimme_parallel(model, expressionRxns, exchange_reactions, num_workers=2):
    """
    Parallelized implementation of the GIMME algorithm across multiple expression samples.

    This function performs the following steps:
      1. Saves the original reaction order.
      2. Uses the provided list of exchange reaction IDs.
      3. Counts how many of these exchange reactions are reversible.
      4. Sets up objective reactions for biomass and ATPase.
      5. Runs FBA for these objectives and updates their lower bounds (20% of max flux).
      6. Updates irreversible exchange reactions using FVA and then splits reversible exchange reactions.
      7. Extracts necessary model parameters (stoichiometric matrix, bounds, reaction and metabolite IDs).
      8. Optionally writes a validated SBML for workers to load; falls back to deep copies if SBML invalid.
      9. Runs GIMME in parallel and aggregates the recombined results.

    Returns path to CSV with aggregated solutions.
    """

    # Save original reaction IDs before any modifications.
    original_rxn_ids = [rxn.id for rxn in model.reactions]
    print(f"Original number of reactions: {len(original_rxn_ids)}")

    print(f"Total exchange reactions in Endothelial Cells: {len(exchange_reactions)}")

    # Count how many exchange reactions are reversible.
    count_reversible = sum(
        model.reactions.get_by_id(rxn).reversibility for rxn in exchange_reactions
    )
    print(f"Total reversible exchange reactions: {count_reversible}")

    # Set up your objective reactions.
    biomass_rxn = model.reactions.get_by_id("MAR13082")  # Biomass reaction
    atpase_rxn = model.reactions.get_by_id("MAR03964")  # ATPase reaction

    # Get and store original objective.
    original_objective = model.objective.expression

    # Perform FBA for biomass and ATPase to determine maximum fluxes.
    model.objective = biomass_rxn
    biomass_max = model.optimize().objective_value
    model.objective = atpase_rxn
    atpase_max = model.optimize().objective_value
    print(f"Max Biomass Flux: {biomass_max}, Max ATPase Flux: {atpase_max} (using FBA)")

    # Set lower bounds (20% of maximum flux) for these reactions.
    model.reactions.get_by_id("MAR03964").lower_bound = float(0.2 * atpase_max)
    model.reactions.get_by_id("MAR13082").lower_bound = float(0.2 * biomass_max)

    # Update irreversible exchange reactions.
    irreversible_fva_filename = os.path.join(
        project_root, "files", "irreversible_rxns_fva.csv"
    )
    model = update_irreversible_exchanges(
        model, exchange_reactions, fva_filename=irreversible_fva_filename
    )

    # Use the first sample to build an expression dictionary for splitting reversible reactions.
    sample0_expr_dict = build_expression_dict_from_original(
        original_rxn_ids, expressionRxns[:, 0]
    )
    reversible_fva_filename = os.path.join(
        project_root, "files", "reversible_rxns_fva.csv"
    )
    model, split_mapping, _ = split_reversible_exchanges(
        model,
        sample0_expr_dict,
        exchange_reactions,
        fva_filename=reversible_fva_filename,
    )

    # After modifications, extract necessary data.
    reaction_ids_modified = [rxn.id for rxn in model.reactions]
    metabolite_ids = [met.id for met in model.metabolites]
    S = create_stoichiometric_matrix(model)
    lb = np.array([float(rxn.lower_bound) for rxn in model.reactions], dtype=np.float64)
    ub = np.array([float(rxn.upper_bound) for rxn in model.reactions], dtype=np.float64)

    # Define the weighted objective function (e.g., 10% Biomass and 90% ATPase).
    biomass_modified_index = reaction_ids_modified.index("MAR13082")
    atpase_modified_index = reaction_ids_modified.index("MAR03964")

    print(
        f"biomass_modified_index: {biomass_modified_index}, atpase_modified_index: {atpase_modified_index}"
    )
    print(f"Total reactions in modified model: {len(reaction_ids_modified)}")

    objectives = [{biomass_modified_index: 0.1}, {atpase_modified_index: 0.9}]
    obj_frac = 1.0

    flux_threshold = 1e-7

    # Prepare an array to hold all the solutions.
    all_solutions = np.zeros((expressionRxns.shape[0], expressionRxns.shape[1]))

    if num_workers == 1:
        # Single processor: run directly in main process to avoid multiprocessing overhead
        print("Running GIMME in single-threaded mode (no multiprocessing).")

        for sample_idx in range(min(10, expressionRxns.shape[1])):
            sample_idx, net_solution = gimme_worker(
                sample_idx,
                expressionRxns[:, sample_idx],
                original_rxn_ids,
                copy.deepcopy(model),  # Use deep copy directly
                reaction_ids_modified,
                metabolite_ids,
                split_mapping,
                S,
                lb,
                ub,
                objectives,
                obj_frac,
                flux_threshold,
            )

            if net_solution is not None:
                all_solutions[:, sample_idx] = net_solution
            else:
                all_solutions[:, sample_idx] = np.zeros(expressionRxns.shape[0])
                print(f"Warning: Sample {sample_idx + 1} failed, filled with zeros.")
    else:
        # Multi-processor: use ProcessPoolExecutor
        print(f"Running GIMME in multi-threaded mode with {num_workers} workers.")

        # Create a spawn-based multiprocessing context
        ctx = multiprocessing.get_context("spawn")

        # Save the modified model to an SBML file so each worker CAN load a fresh copy.
        modified_model_sbml = os.path.join(
            project_root, "files", "modified_model_for_parallel.sbml"
        )

        use_sbml = False
        try:
            cobra.io.write_sbml_model(model, modified_model_sbml)
            try:
                valid_model, sbml_errors = cobra.io.sbml.validate_sbml_model(
                    modified_model_sbml
                )
                if sbml_errors:
                    print("SBML validation reported errors:", sbml_errors)
                    raise Exception("SBML validation failed")
                else:
                    use_sbml = True
                    print(f"Wrote and validated SBML model at {modified_model_sbml}")
            except Exception as ve:
                print(f"SBML validation failed: {ve}")
        except Exception as write_err:
            print(f"Failed to write SBML model to {modified_model_sbml}: {write_err}")

        if not use_sbml:
            print(
                "Falling back to passing deep-copied model objects to workers (no SBML)."
            )

        # Parallelize over samples using ProcessPoolExecutor.
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:

            # Submit a job for each sample (limited to the first 10 samples for debugging).
            futures = {
                executor.submit(
                    gimme_worker,
                    sample_idx,
                    expressionRxns[:, sample_idx],
                    original_rxn_ids,
                    (modified_model_sbml if use_sbml else copy.deepcopy(model)),
                    reaction_ids_modified,
                    metabolite_ids,
                    split_mapping,
                    S,
                    lb,
                    ub,
                    objectives,
                    obj_frac,
                    flux_threshold,
                ): sample_idx
                for sample_idx in range(min(10, expressionRxns.shape[1]))
            }

            # Collect results as they complete.
            for future in futures:
                sample_idx, net_solution = future.result()
                if net_solution is not None:
                    all_solutions[:, sample_idx] = net_solution
                else:
                    all_solutions[:, sample_idx] = np.zeros(expressionRxns.shape[0])
                    print(
                        f"Warning: Sample {sample_idx + 1} failed, filled with zeros."
                    )

    # Save the aggregated solutions.
    output_file = os.path.join(project_root, "files", "allsolutions_gimme_parallel.csv")
    pd.DataFrame(all_solutions).to_csv(output_file, index=False, header=False)

    # Reset the objective to its original state.
    model.objective = original_objective
    return output_file


def main():
    expressionRxns = pd.read_csv(
        os.path.join(project_root, "files", "expressionRxns.csv"), header=None
    ).values
    exchange_rs = (
        pd.read_csv(os.path.join(project_root, "files", "common_rs.txt"), header=None)
        .values.flatten()
        .tolist()
    )

    model_path = os.path.join(project_root, "models", "model_full_THG.xml")
    model = cobra.io.read_sbml_model(model_path)

    output_file = gimme_parallel(model, expressionRxns, exchange_rs, num_workers=1)
    print("GIMME parallel run complete, results saved to:", output_file)


if __name__ == "__main__":
    main()
