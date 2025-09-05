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
            extended_scores.append(original_expr_dict[rxn.id])
        else:
            if rxn.id.endswith("_fwd") or rxn.id.endswith("_rev"):
                base_id = rxn.id.rsplit("_", 1)[0]
                extended_scores.append(original_expr_dict.get(base_id, 0.0))
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
        fva_results = flux_variability_analysis(
            model, reaction_list=reversible_exchanges
        )
        fva_results.to_csv(fva_filename)

    for rxn in reversible_exchanges:
        rxn = model.reactions.get_by_id(rxn)

        # Retrieve FVA results for this reaction, skip if not available.
        if rxn.id not in fva_results.index:
            print(f"Warning: FVA results not found for reaction {rxn.id}, skipping...")
            continue

        fva_max = fva_results.loc[rxn.id, "maximum"]
        fva_min = fva_results.loc[rxn.id, "minimum"]

        # Set forward lower bound if the reaction is capable of carrying positive flux.
        forward_lb = min_bound if fva_max > 0 else 0.0
        # Set reverse lower bound if the reaction is capable of carrying negative flux.
        reverse_lb = min_bound if fva_min < 0 else 0.0
        # For the reverse reaction, set the upper bound to the absolute value of the minimum flux (if negative).
        reverse_ub = abs(fva_min) if fva_min < 0 else 0.0

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

        # If the FVA maximum flux is > 0, update the lower bound to min_bound.
        # But ensure the new lower bound doesn't exceed the current upper bound.
        # Also skip if the upper bound is 0 (reaction is blocked).
        if max_flux > 0 and rxn.upper_bound > 0:
            new_lower_bound = min(min_bound, rxn.upper_bound)
            if new_lower_bound < rxn.upper_bound:  # Only set if it's a valid bound
                rxn.lower_bound = new_lower_bound

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
    model,
    reaction_ids_modified,
    metabolite_ids,
    split_mapping,
    S,
    lb,
    ub,
    objectives,
    obj_frac,
    flux_threshold,
    solver=None,
):
    """
    Worker function to run the GIMME algorithm in parallel.

    Parameters:
        sample_idx (int): Index of the sample.
        sample_expr (array): Expression vector for the sample.
        original_rxn_ids (list): Original reaction IDs.
        model (cobra.Model): Modified metabolic model.
        reaction_ids_modified (list): Reaction IDs after model modifications.
        metabolite_ids (list): List of metabolite IDs.
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

    print(f"Running GIMME for sample {sample_idx + 1}...")

    # Build the sample-specific expression dictionary.
    expr_dict = build_expression_dict_from_original(original_rxn_ids, sample_expr)

    # Debug: Check expression values
    non_neg_one_values = sum(1 for v in sample_expr if v != -1)
    print(
        f"Debug: Sample {sample_idx + 1} has {non_neg_one_values} non-(-1) expression values out of {len(sample_expr)}"
    )
    if non_neg_one_values > 0:
        non_neg_one_sample = [v for v in sample_expr if v != -1][
            :5
        ]  # First 5 non-(-1) values
        print(f"Debug: First few non-(-1) values: {non_neg_one_sample}")
    else:
        print(
            f"Debug: All expression values are -1 for sample {sample_idx + 1}. This indicates a problem with gene expression processing."
        )
        # Return early if no valid expression data
        return sample_idx, np.zeros(len(original_rxn_ids))

    # Compute the extended scores for the modified model.
    scores = get_extended_scores(model, expr_dict)
    print(f"Number of reactions (including splits): {len(scores)}")
    print(
        f"For threshold {flux_threshold}: active reactions = {sum((scores > flux_threshold) & (scores > -1))}. Number of uncertain reactions = {sum(scores == -1)}"
    )

    # Set up the GIMME properties.

    properties = GIMMEProperties(
        exp_vector=scores,
        obj_frac=obj_frac,
        objectives=objectives,
        preprocess=False,
        flux_threshold=flux_threshold,
    )

    # Create and run the GIMME algorithm.
    gimme_instance = GIMME(S=S, lb=lb, ub=ub, properties=properties, solver=solver)
    gimme_instance.run()

    # Extract the raw solution and recombine it to the original reaction order.
    solution = gimme_instance.sol.__dict__["_Solution__value_map"]
    reaction_ids_modified = ["R" + str(i) for i in range(S.shape[1])]
    net_solution = recombine_solution(
        solution, original_rxn_ids, split_mapping, reaction_ids_modified
    )
    print(
        f"Sample {sample_idx + 1} objective value: {gimme_instance.sol.objective_value}. Active reactions: {sum(net_solution > flux_threshold)}"
    )

    # debug
    # print("solution: ", solution)
    print("split_mapping: ", split_mapping)
    print("net_solution: ", net_solution)

    return sample_idx, net_solution


# Main parallelized GIMME function.
def gimme_parallel(
    model, expressionRxns, exchange_reactions, num_workers=2, solver=None
):
    """
    Parallelized implementation of the GIMME algorithm across multiple expression samples.

    This function performs the following steps:
      1. Saves the original reaction order.
      2. Uses the provided list of exchange reaction IDs.
      3. Counts how many of these exchange reactions are reversible.
      4. Sets up objective reactions for biomass and ATPase.
      5. Runs FBA to determine maximum fluxes for these objective reactions and updates their lower bounds (20% of max flux).
      6. Updates irreversible exchange reactions using FVA and then splits reversible exchange reactions
         (using FVA results) so that they can be handled as two separate (forward and reverse) reactions.
      7. Extracts necessary model parameters (e.g., stoichiometric matrix, bounds, reaction and metabolite IDs).
      8. Runs the GIMME algorithm in parallel on each expression sample using a ProcessPoolExecutor.
      9. Aggregates the flux solutions from all samples and saves them to a CSV file.
      10. Resets the model's objective to its original state.

    Parameters:
        model (cobra.Model): The metabolic model.
        expressionRxns (np.array): 2D array of expression values (rows: reactions, columns: samples).
        exchange_reactions (list or np.array): List of exchange reaction IDs to process (cell type specific exchange rs).
        num_workers (int): Number of parallel workers to use.

    Returns:
        str: Path to the output CSV file containing the aggregated GIMME solutions.
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
    model.reactions.get_by_id("MAR03964").lower_bound = 0.2 * atpase_max
    model.reactions.get_by_id("MAR13082").lower_bound = 0.2 * biomass_max

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
    lb = np.array([rxn.lower_bound for rxn in model.reactions])
    ub = np.array([rxn.upper_bound for rxn in model.reactions])

    # Define the weighted objective function (e.g., 10% Biomass and 90% ATPase).

    objectives = [{biomass_rxn.id: 0.1}, {atpase_rxn.id: 0.9}]
    obj_frac = np.array([1.0])

    biomass_index = original_rxn_ids.index("MAR13082")
    atpase_index = original_rxn_ids.index("MAR03964")
    # print("original_rxn_ids: ", original_rxn_ids)

    print(f"biomass_index: {biomass_index}, atpase_index: {atpase_index}")  # debug
    objectives = [{biomass_index: 0.1}, {atpase_index: 0.9}]

    flux_threshold = 1e-7

    # Prepare an array to hold all the solutions.
    all_solutions = np.zeros((expressionRxns.shape[0], expressionRxns.shape[1]))

    # Process samples in smaller batches to avoid memory issues
    batch_size = min(2, expressionRxns.shape[1])  # Process max 2 samples at a time
    num_samples = expressionRxns.shape[1]

    print(f"Processing {num_samples} samples in batches of {batch_size}")

    # Create a spawn-based multiprocessing context
    ctx = multiprocessing.get_context("spawn")

    # Process samples in batches
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        print(f"Processing batch: samples {batch_start + 1} to {batch_end}")

        # Use single-threaded processing to avoid memory issues
        # with ProcessPoolExecutor(
        #     max_workers=min(num_workers, batch_size), mp_context=ctx
        # ) as executor:

        # Process samples sequentially for now to avoid memory issues
        for sample_idx in range(batch_start, batch_end):
            print(f"Processing sample {sample_idx + 1}")
            sample_idx, net_solution = gimme_worker(
                sample_idx,
                expressionRxns[:, sample_idx],
                original_rxn_ids,
                model,
                reaction_ids_modified,
                metabolite_ids,
                split_mapping,
                S,
                lb,
                ub,
                objectives,
                obj_frac,
                flux_threshold,
                solver=solver,
            )
            all_solutions[:, sample_idx] = net_solution

    # Save the aggregated solutions.
    output_file = os.path.join(project_root, "files", "allsolutions_gimme_parallel.csv")
    pd.DataFrame(all_solutions).to_csv(output_file, index=False, header=False)

    # Reset the objective to its original state.
    model.objective = original_objective
    return output_file


# Example main function to run the parallelized GIMME.
def main():
    # model_path = os.path.join('transcriptomics', 'model_full_THG.xml')
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
