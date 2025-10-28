import os
import re
import copy
import numpy as np
import pandas as pd
import cobra
from cobra.io import write_sbml_model, save_json_model

from troppo.omics.readers.generic import TabularReader
from troppo.methods_wrappers import ModelBasedWrapper, ReconstructionWrapper
from troppo.omics.integration import ContinuousScoreIntegrationStrategy
from troppo.methods.reconstruction.tINIT import tINIT, tINITProperties

try:
    from multiprocess import Pool

    USE_MULTIPROCESS = True
except ImportError:
    from multiprocessing import Pool

    USE_MULTIPROCESS = False

NUM_CORES = 2

# Toggle: if True, combine all samples by summing expression across columns and run a single tINIT.
# If False, run tINIT per sample.
COMBINE_SAMPLES = True
COMBINED_DEBUG = True  # Print extra diagnostics when combining samples

# Reactions to always keep when building a reduced model (helps feasibility)
# Adjust as needed for your model/objectives.
PROTECTED_REACTIONS = [
    "MAR13082",  # Biomass
    "MAR03964",  # ATP maintenance / ATPase
]


# --- GPR parsing cleanup (as in tutorial) ---
patt = re.compile("__COBAMPGPRDOT__[0-9]{1}")
replace_alt_transcripts = lambda x: patt.sub("", x)  # noqa: E731


def score_apply(reaction_map_scores):
    """Replace None/NaN with 0.0 to meet tINIT expectations.

    tINIT requires a reaction->score mapping with numeric values.
    """
    return {
        k: 0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v
        for k, v in reaction_map_scores.items()
    }


def run_tinit_for_sample(
    sample_idx, reactions_scores_list, S, lb, ub, reaction_ids, solver="GUROBI"
):
    """Run tINIT for a single sample.

    Inputs:
    - reactions_scores_list: list[float] aligned with reaction_ids/model order
    - S, lb, ub: model matrices/vectors from the wrapper
    - reaction_ids: list[str] reaction ids in the same order as S columns

    Returns:
    - selection_vector: np.ndarray[int] with 1 for kept reactions, else 0
    - selected_ids: list[str] of selected reaction ids
    """
    print(f"\n=== Running tINIT for sample {sample_idx + 1} ===")
    # Ensure correct length
    if len(reactions_scores_list) != len(reaction_ids):
        print(
            f"Warning: reactions_scores length {len(reactions_scores_list)} != number of reactions {len(reaction_ids)}; adjusting"
        )
        scores = list(reactions_scores_list)[: len(reaction_ids)] + [0.0] * max(
            0, len(reaction_ids) - len(reactions_scores_list)
        )
    else:
        scores = list(reactions_scores_list)

    # Debug: summarize the score vector
    try:
        arr = np.asarray(scores, dtype=float)
        nz = int(np.sum(arr > 0))
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        vmean = float(np.mean(arr))
        vmed = float(np.median(arr))
        print(
            f"Scores summary -> nnz={nz}/{len(arr)}, min={vmin:.3g}, median={vmed:.3g}, mean={vmean:.3g}, max={vmax:.3g}"
        )
    except Exception:
        pass

    properties = tINITProperties(reactions_scores=scores, solver=solver)
    algo = tINIT(S=S, lb=lb, ub=ub, properties=properties)
    result = algo.run()

    # Try to interpret result and build a selection vector
    selection = np.zeros(len(reaction_ids), dtype=int)
    selected_ids = []
    try:
        # Case 1: list/array of indices
        if isinstance(result, (list, tuple, np.ndarray)) and all(
            isinstance(x, (int, np.integer)) for x in result
        ):
            for i in result:
                if 0 <= int(i) < len(selection):
                    selection[int(i)] = 1
            selected_ids = [reaction_ids[int(i)] for i in np.nonzero(selection)[0]]
        # Case 2: dict mapping reaction_id -> bool/score
        elif isinstance(result, dict):
            for rid, val in result.items():
                if rid in reaction_ids and (
                    bool(val) or (isinstance(val, (int, float)) and float(val) != 0.0)
                ):
                    selection[reaction_ids.index(rid)] = 1
            selected_ids = [reaction_ids[i] for i in np.nonzero(selection)[0]]
        else:
            # Unknown format; try to read selection from algo if available
            possible_attrs = [
                "selected_rxns",
                "selected_reactions",
                "solution",
                "sol",
            ]
            for attr in possible_attrs:
                if hasattr(algo, attr):
                    sel = getattr(algo, attr)
                    if isinstance(sel, (list, tuple, np.ndarray)):
                        for i in sel:
                            if isinstance(i, (int, np.integer)) and 0 <= int(i) < len(
                                selection
                            ):
                                selection[int(i)] = 1
                            elif isinstance(i, str) and i in reaction_ids:
                                selection[reaction_ids.index(i)] = 1
                        break
        print(
            f"tINIT sample {sample_idx + 1}: selected {int(selection.sum())}/{len(selection)} reactions"
        )
        if selection.sum() > 0:
            print(
                "  examples:",
                ", ".join(
                    [
                        rid
                        for rid in reaction_ids[:5]
                        if selection[reaction_ids.index(rid)] == 1
                    ][:5]
                ),
            )
    except Exception as e:
        print(f"Warning: could not parse tINIT result for sample {sample_idx + 1}: {e}")

    if not selected_ids:
        selected_ids = [reaction_ids[i] for i in np.nonzero(selection)[0]]

    return selection, selected_ids


def build_and_save_reduced_model(
    model,
    keep_reaction_ids,
    project_root,
    model_path,
    suffix,
    protected_reaction_ids=None,
    include_boundary=True,
):
    """Build a reduced model keeping only keep_reaction_ids, then save SBML and JSON.

    - model: cobra.Model (source model)
    - keep_reaction_ids: list[str] reaction IDs to keep
    - project_root: base path to repo
    - model_path: original model path (used for naming)
    - suffix: string suffix to append to base filename
    """
    # Filter to reactions that exist in the model
    keep_set = set(keep_reaction_ids)
    # Add protected reactions (e.g., biomass, ATPase)
    if protected_reaction_ids:
        keep_set.update(protected_reaction_ids)
    # Optionally include all boundary reactions (exchanges, sinks, demands)
    if include_boundary:
        try:
            boundary_ids = [r.id for r in model.boundary]
            keep_set.update(boundary_ids)
        except Exception:
            pass
    all_ids = {r.id for r in model.reactions}
    keep_set = keep_set.intersection(all_ids)

    model_red = model.copy()
    to_remove = [r for r in model_red.reactions if r.id not in keep_set]
    if to_remove:
        model_red.remove_reactions(to_remove)

    # Remove orphan metabolites and genes
    mets_to_remove = [met for met in model_red.metabolites if len(met.reactions) == 0]
    if mets_to_remove:
        model_red.remove_metabolites(mets_to_remove)
    genes_to_remove = [g for g in model_red.genes if len(g.reactions) == 0]
    for g in genes_to_remove:
        try:
            model_red.genes.remove(g)
        except Exception:
            pass

    base_name = os.path.splitext(os.path.basename(model_path))[0]
    out_xml = os.path.join(
        project_root, "models", f"{base_name}_tinit_reduced_{suffix}.xml"
    )
    out_json = os.path.join(
        project_root, "models", f"{base_name}_tinit_reduced_{suffix}.json"
    )
    write_sbml_model(model_red, out_xml)
    try:
        save_json_model(model_red, out_json)
    except Exception as e:
        print(f"Warning: failed to save JSON for reduced model {suffix}: {e}")
    print(
        f"Reduced model ({suffix}) saved: reactions={len(model_red.reactions)}, metabolites={len(model_red.metabolites)}, genes={len(model_red.genes)}"
    )
    print("  SBML:", out_xml)
    print("  JSON:", out_json)
    # Quick feasibility check
    try:
        sol = model_red.optimize()
        print(
            f"  Feasibility check: status={sol.status}, objective={sol.objective_value}"
        )
    except Exception as e:
        print(f"  Feasibility check failed: {e}")


def main():
    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    # Pick the same model used in other tutorials
    model_path = os.path.join(project_root, "models", "endoA_250919.xml")
    # Use reaction-level expression matrix (rows=reaction ids, columns=samples)
    expression_rxns_path = os.path.join(project_root, "files", "expressionRxns.csv")

    # Load model and expression data
    model = cobra.io.read_sbml_model(model_path)

    # Optional: clean GPRs as in tutorial (remove COBAMPGPRDOT artifacts)
    for rxn in model.reactions:
        rule = rxn.gene_reaction_rule
        if rule and isinstance(rule, str):
            rxn.gene_reaction_rule = replace_alt_transcripts(rule)

    # Wrap model for Troppo to access S/lb/ub
    model_wrapper = ReconstructionWrapper(
        model=model, ttg_ratio=9999, gpr_gene_parse_function=replace_alt_transcripts
    )

    reaction_ids = model_wrapper.model_reader.r_ids
    S = model_wrapper.S
    lb = model_wrapper.lb
    ub = model_wrapper.ub

    # Read reaction-level expression.
    # Support two formats:
    # 1) ID-based: first column has reaction IDs, remaining columns are samples
    # 2) Positional: no IDs; rows follow the model reaction order
    raw_df = pd.read_csv(expression_rxns_path, header=None)

    # Try to detect ID-based format
    id_based = False
    try:
        first_col = raw_df.iloc[:, 0]
        if first_col.dtype == object:
            rid_set = set(reaction_ids)
            id_matches = sum(
                1 for x in first_col if isinstance(x, str) and x in rid_set
            )
            coverage_est = id_matches / max(1, len(reaction_ids))
            # Heuristic: consider ID-based if we have a noticeable number of matches
            id_based = id_matches > 0 and (coverage_est > 0.01 or id_matches > 50)
    except Exception:
        id_based = False

    if id_based:
        expr_df = raw_df.set_index(0)
        expr_df = expr_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        num_samples = expr_df.shape[1]
        n_rows = expr_df.shape[0]
        if n_rows != len(reaction_ids):
            print(
                f"Warning: ID-based file has {n_rows} rows but model has {len(reaction_ids)} reactions; missing rows will map to 0 if absent."
            )
    else:
        # Positional mapping mode
        expr_mat = raw_df.values
        # Ensure numeric
        expr_mat = (
            pd.DataFrame(expr_mat)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .values
        )
        n_rows, num_samples = expr_mat.shape
        if n_rows != len(reaction_ids):
            print(
                f"Warning: positional file rows ({n_rows}) != number of model reactions ({len(reaction_ids)}). Will truncate/pad with zeros."
            )
            if n_rows > len(reaction_ids):
                expr_mat = expr_mat[: len(reaction_ids), :]
            else:
                pad = np.zeros((len(reaction_ids) - n_rows, num_samples))
                expr_mat = np.vstack([expr_mat, pad])

    num_rxns = len(reaction_ids)

    # Build per-sample reaction score lists aligned to reaction_ids
    def reaction_scores_for_col(series):
        # series index are reaction IDs in file
        return [
            (
                0.0
                if (rid not in series.index or pd.isna(series.loc[rid]))
                else float(series.loc[rid])
            )
            for rid in reaction_ids
        ]

    if id_based:
        reactions_scores_per_sample = [
            reaction_scores_for_col(expr_df.iloc[:, i]) for i in range(num_samples)
        ]
    else:
        # Positional: use columns directly aligned to reaction order
        reactions_scores_per_sample = [
            expr_mat[:, i].astype(float).tolist() for i in range(num_samples)
        ]

    if COMBINE_SAMPLES:
        # Combine samples: sum expression across columns to produce one aggregate vector
        print(
            f"Combining {num_samples} samples by summation and running a single tINIT."
        )
        if COMBINED_DEBUG:
            if id_based:
                print(
                    f"expr_df shape: {expr_df.shape} (rows=reactions in file, cols=samples)"
                )
                match_count = len(set(expr_df.index).intersection(set(reaction_ids)))
                print(
                    f"Mapping coverage (file idx ∩ model reactions): {match_count}/{len(reaction_ids)} = {match_count/len(reaction_ids):.1%}"
                )
                missing_ids = [rid for rid in reaction_ids if rid not in expr_df.index][
                    :10
                ]
                if missing_ids:
                    print(
                        "Example reaction IDs missing from file index:",
                        ", ".join(missing_ids[:10]),
                    )
                try:
                    sample_sums = expr_df.sum(axis=0).to_numpy()
                    print(
                        "Sample column sums (first 10):",
                        ", ".join([f"{x:.3g}" for x in sample_sums[:10]]),
                    )
                except Exception:
                    pass
            else:
                print(
                    f"Positional mode: matrix shape {expr_mat.shape} (rows=model reaction order, cols=samples)"
                )
                if expr_mat.shape[0] != len(reaction_ids):
                    print(
                        f"Note: rows != model reactions ({expr_mat.shape[0]} vs {len(reaction_ids)}); truncated/padded to align."
                    )
                sample_sums = np.sum(expr_mat, axis=0)
                print(
                    "Sample column sums (first 10):",
                    ", ".join([f"{x:.3g}" for x in sample_sums[:10]]),
                )

        if id_based:
            aggregate_series = expr_df.sum(axis=1)
            aggregate_scores = [
                (
                    0.0
                    if (
                        rid not in aggregate_series.index
                        or pd.isna(aggregate_series.loc[rid])
                    )
                    else float(aggregate_series.loc[rid])
                )
                for rid in reaction_ids
            ]
        else:
            aggregate_scores = np.sum(expr_mat, axis=1).astype(float).tolist()
        if COMBINED_DEBUG:
            agg_arr = np.asarray(aggregate_scores, dtype=float)
            nz = int(np.sum(agg_arr > 0))
            zeros = int(np.sum(agg_arr == 0))
            qs = np.quantile(agg_arr, [0.0, 0.25, 0.5, 0.75, 1.0])
            print(
                f"Aggregate scores -> nnz={nz}, zeros={zeros}, min/25/50/75/max = "
                + ", ".join([f"{q:.3g}" for q in qs])
            )
            # Top reactions by aggregate score
            pairs = list(zip(reaction_ids, agg_arr))
            pairs.sort(key=lambda x: x[1], reverse=True)
            print(
                "Top 10 reactions by aggregate score:",
                ", ".join([f"{rid}:{val:.3g}" for rid, val in pairs[:10]]),
            )
        sel_vec, sel_ids = run_tinit_for_sample(
            0, aggregate_scores, S, lb, ub, reaction_ids, "GUROBI"
        )

        # Save combined outputs
        out_matrix = os.path.join(
            project_root, "files", "tinit_selection_matrix_combined.csv"
        )
        pd.DataFrame(sel_vec.reshape(-1, 1), index=reaction_ids).to_csv(
            out_matrix, header=False
        )
        print(f"\nSaved combined tINIT selection vector to: {out_matrix}")

        out_ids = os.path.join(
            project_root, "files", "tinit_selected_rxns_combined.txt"
        )
        with open(out_ids, "w") as fh:
            for rid in sel_ids:
                fh.write(f"{rid}\n")
        print(f"Saved combined selected reaction IDs to: {out_ids}")
        # Build and save combined reduced model
        build_and_save_reduced_model(
            model,
            sel_ids,
            project_root,
            model_path,
            suffix="combined",
            protected_reaction_ids=PROTECTED_REACTIONS,
            include_boundary=True,
        )
    else:
        # tINIT per sample (optionally in parallel)
        selections = np.zeros((num_rxns, num_samples), dtype=int)
        selected_ids_per_sample = []

        args = [
            (i, reactions_scores_per_sample[i], S, lb, ub, reaction_ids, "GUROBI")
            for i in range(num_samples)
        ]

        if NUM_CORES > 1:
            print(
                f"DEBUG: Starting tINIT tutorial with {num_samples} samples using {'multiprocess' if USE_MULTIPROCESS else 'multiprocessing'} on {NUM_CORES} cores"
            )
            with Pool(processes=NUM_CORES) as pool:
                results = pool.starmap(run_tinit_for_sample, args)
            for j, (sel_vec, sel_ids) in enumerate(results):
                selections[:, j] = sel_vec
                selected_ids_per_sample.append(sel_ids)
        else:
            for j in range(num_samples):
                sel_vec, sel_ids = run_tinit_for_sample(*args[j])
                selections[:, j] = sel_vec
                selected_ids_per_sample.append(sel_ids)

        # Save outputs
        out_matrix = os.path.join(project_root, "files", "tinit_selection_matrix.csv")
        pd.DataFrame(selections, index=reaction_ids).to_csv(out_matrix, header=False)
        print(f"\nSaved tINIT selection matrix to: {out_matrix}")

        # Also write per-sample selected reaction IDs (one file per sample for convenience)
        for j, sel_ids in enumerate(selected_ids_per_sample):
            out_ids = os.path.join(
                project_root, "files", f"tinit_selected_rxns_sample_{j+1}.txt"
            )
            with open(out_ids, "w") as fh:
                for rid in sel_ids:
                    fh.write(f"{rid}\n")
            print(f"Saved sample {j+1} selected reaction IDs to: {out_ids}")
        # Build and save reduced model per sample
        for j, sel_ids in enumerate(selected_ids_per_sample, start=1):
            build_and_save_reduced_model(
                model,
                sel_ids,
                project_root,
                model_path,
                suffix=f"sample_{j}",
                protected_reaction_ids=PROTECTED_REACTIONS,
                include_boundary=True,
            )

    print(
        "\nIf you want to build reduced models, adapt one of the existing reducers (e.g., create_reduced_model_from_fastcore.py) to take the tINIT selection list."
    )


if __name__ == "__main__":
    main()
