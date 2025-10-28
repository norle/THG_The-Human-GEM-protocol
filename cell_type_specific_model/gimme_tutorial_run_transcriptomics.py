import pandas as pd
import numpy as np
import cobra
import re
from troppo.omics.readers.generic import TabularReader
from troppo.methods_wrappers import ModelBasedWrapper, ReconstructionWrapper
from troppo.omics.integration import ContinuousScoreIntegrationStrategy
from troppo.methods.reconstruction.gimme import GIMME, GIMMEProperties
import os
import copy

try:
    from multiprocess import Pool

    USE_MULTIPROCESS = True
except ImportError:
    from multiprocessing import Pool

    USE_MULTIPROCESS = False

NUM_CORES = 2


def score_apply(reaction_map_scores):
    return {k: 0 if v is None else v for k, v in reaction_map_scores.items()}


def run_gimme_for_sample(
    sample_idx,
    exp_vector,
    S,
    lb,
    ub,
    reaction_ids,
    metabolite_ids,
    idx_objective,
    num_rxns,
):
    """Run GIMME for a single sample using a precomputed expression vector.

    exp_vector: list or 1D-array aligned to reaction_ids order (same order as model)
    S, lb, ub, reaction_ids, metabolite_ids: model matrices/vectors (picklable)
    """
    print(f"\n=== Running GIMME for sample {sample_idx + 1} ===")

    # Ensure exp_vector length matches reaction count
    if len(exp_vector) != len(reaction_ids):
        print(
            f"Warning: exp_vector length {len(exp_vector)} != number of reactions {len(reaction_ids)}; filling missing with zeros"
        )
        # pad or truncate as needed
        vec = list(exp_vector)[: len(reaction_ids)] + [0.0] * max(
            0, len(reaction_ids) - len(exp_vector)
        )
    else:
        vec = list(exp_vector)

    properties = GIMMEProperties(
        exp_vector=vec,
        obj_frac=0.8,
        objectives=[{idx_objective: 1}],
        preprocess=True,
        flux_threshold=0.8,
        solver="GUROBI",
        reaction_ids=reaction_ids,
        metabolite_ids=metabolite_ids,
    )

    gimme = GIMME(S=S, lb=lb, ub=ub, properties=properties)
    gimme_run = gimme.run()
    flux_solution = getattr(getattr(gimme, "sol", None), "__dict__", {}).get(
        "_Solution__value_map", None
    )
    print(f"GIMME run result (indices or IDs): {gimme_run}")
    # Determine active reactions. Prefer gimme_run (indices), fallback to non-zero fluxes.
    active_ids = []
    if gimme_run is not None:
        try:
            # assume gimme_run is iterable of indices
            active_ids = [
                reaction_ids[i] for i in gimme_run if 0 <= i < len(reaction_ids)
            ]
        except Exception:
            # if gimme_run already contains reaction ids
            try:
                active_ids = [rid for rid in gimme_run if rid in reaction_ids]
            except Exception:
                active_ids = []
    elif flux_solution is not None:
        # flux_solution maps reaction_id -> flux
        active_ids = [
            rid for rid in reaction_ids if abs(flux_solution.get(rid, 0.0)) > 1e-9
        ]

    inactive_count = len(reaction_ids) - len(active_ids)
    # print concise summary
    sample_label = sample_idx + 1
    print(
        f"Sample {sample_label}: active reactions = {len(active_ids)}, inactive = {inactive_count}"
    )
    if len(active_ids) > 0:
        print("  examples active:", ", ".join(active_ids[:5]))
    else:
        print("  no active reactions identified")

    if flux_solution is not None:
        return [flux_solution.get(rxn_id, 0.0) for rxn_id in reaction_ids]
    else:
        print(f"Warning: Sample {sample_idx + 1} failed, filled with zeros.")
        return [0.0] * num_rxns


def main():
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    model_path = os.path.join(project_root, "models", "endoA_250904_clean.xml")
    expression_rxns_path = os.path.join(project_root, "files", "expressionRxns.csv")

    # Load model and expression data
    model = cobra.io.read_sbml_model(model_path)
    # expressionRxns.csv has reaction ids as rows and samples in columns
    # Expect first column to be reaction ids if file came from transcriptomics pipeline
    expression_data = pd.read_csv(expression_rxns_path, header=None, index_col=0)

    expression_data = expression_data.iloc[:, :3]
    # --- Model Wrapper ---
    # reactions already mapped in expression CSV, but some GPR strings contain
    # leading underscores (e.g. '_ENSG...') which cause cobamp normalizer warnings.
    # Apply a conservative in-place cleanup: strip leading underscores only from
    # gene tokens (keeps other text unchanged).
    for rxn in model.reactions:
        rule = rxn.gene_reaction_rule
        if rule and isinstance(rule, str):
            # only remove leading underscores from tokens that start with underscore
            cleaned = re.sub(r"\b_+(ENSG[0-9A-Za-z_:-]+)\b", r"\1", rule)
            cleaned = re.sub(r"\b_+([A-Za-z0-9]+)\b", r"\1", cleaned)
            # normalize whitespace
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            rxn.gene_reaction_rule = cleaned

    model_wrapper = ReconstructionWrapper(model=model, ttg_ratio=9999)

    biomass_rxn_id = "MAR13082"
    idx_objective = model_wrapper.model_reader.r_ids.index(biomass_rxn_id)
    reaction_ids = model_wrapper.model_reader.r_ids
    metabolite_ids = model_wrapper.model_reader.m_ids
    num_rxns = len(reaction_ids)

    # Build per-sample expression vectors aligned to model reaction order.
    # If a reaction from the model is missing in the CSV, fill with 0.0 (treated as missing)
    # expression_data rows should be reaction ids; convert to float and map.
    expr_df = expression_data.astype(float)
    num_samples = expr_df.shape[1]
    all_solutions = np.zeros((num_rxns, num_samples))

    exp_vectors = []
    for i in range(num_samples):
        col_series = expr_df.iloc[:, i]
        # map according to reaction_ids order
        vec = [
            (
                0.0
                if (rid not in col_series.index or pd.isna(col_series.loc[rid]))
                else float(col_series.loc[rid])
            )
            for rid in reaction_ids
        ]
        exp_vectors.append(vec)

    # Prepare arguments for multiprocessing: pass precomputed exp_vector and model matrices
    S = model_wrapper.S
    lb = model_wrapper.lb
    ub = model_wrapper.ub
    args = [
        (
            i,
            exp_vectors[i],
            S,
            lb,
            ub,
            reaction_ids,
            metabolite_ids,
            idx_objective,
            num_rxns,
        )
        for i in range(num_samples)
    ]

    print(
        f"DEBUG: About to start multiprocessing with {num_samples} samples", flush=True
    )
    print(
        f"Using {'multiprocess' if USE_MULTIPROCESS else 'multiprocessing'} with {NUM_CORES} cores",
        flush=True,
    )
    print(f"DEBUG: Starting Pool now", flush=True)
    with Pool(processes=NUM_CORES) as pool:
        results = pool.starmap(run_gimme_for_sample, args)

    for sample_idx, solution in enumerate(results):
        all_solutions[:, sample_idx] = solution

    # Save aggregated solutions as CSV
    output_file = os.path.join(project_root, "files", "allsolutions_gimme_tutorial.csv")
    pd.DataFrame(all_solutions, index=model_wrapper.model_reader.r_ids).to_csv(
        output_file, header=False
    )
    print(f"\nAggregated flux solutions saved to: {output_file}")

    print("\nRun the following script to create reduced models:")
    print("python create_reduced_model_from_gimme.py")


if __name__ == "__main__":
    main()
