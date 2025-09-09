import pandas as pd
import numpy as np
import cobra
import re
from troppo.omics.readers.generic import TabularReader
from troppo.methods_wrappers import ModelBasedWrapper, ReconstructionWrapper
from troppo.omics.integration import ContinuousScoreIntegrationStrategy
from troppo.methods.reconstruction.imat import IMAT, IMATProperties
import os
import copy
import time

try:
    from multiprocess import Pool

    USE_MULTIPROCESS = True
except ImportError:
    from multiprocessing import Pool

    USE_MULTIPROCESS = False

NUM_CORES = 30


def score_apply(reaction_map_scores):
    return {k: 0 if v is None else v for k, v in reaction_map_scores.items()}


def run_imat_for_sample(
    sample_idx, expression_data, model_wrapper, exp_thresholds, num_rxns
):
    start_time = time.time()
    print(f"\n=== Running iMAT for sample {sample_idx + 1} ===")
    sample_df = expression_data.iloc[:, [sample_idx]]
    omics_container = TabularReader(
        path_or_df=sample_df,
        nomenclature="ensembl_id",  # adjust if needed
        omics_type="transcriptomics",
        sample_in_rows=False,
    ).to_containers()
    single_sample = omics_container[0]
    data_map = single_sample.get_integrated_data_map(
        model_reader=model_wrapper.model_reader, and_func=min, or_func=sum
    )
    continuous_integration = ContinuousScoreIntegrationStrategy(score_apply=score_apply)
    scores = continuous_integration.integrate(data_map=data_map)

    # Create the properties for the iMAT algorithm
    properties = IMATProperties(exp_vector=scores, exp_thresholds=exp_thresholds)

    # Run the iMAT algorithm
    imat = IMAT(
        S=model_wrapper.S,
        lb=model_wrapper.lb,
        ub=model_wrapper.ub,
        properties=properties,
    )

    solve_start_time = time.time()
    active_reactions = imat.run()
    solve_time = time.time() - solve_start_time
    total_time = time.time() - start_time

    print(
        f"Sample {sample_idx + 1}: {len(active_reactions)} active reactions | "
        f"Solve time: {solve_time:.2f}s | Total time: {total_time:.2f}s"
    )

    # Create binary solution vector (1 for active reactions, 0 for inactive)
    solution = np.zeros(num_rxns)
    if active_reactions is not None:
        for rxn_idx in active_reactions:
            if rxn_idx < num_rxns:  # Safety check
                solution[rxn_idx] = 1.0
    else:
        print(f"Warning: Sample {sample_idx + 1} failed, filled with zeros.")

    return solution, solve_time


def main():
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    model_path = os.path.join(project_root, "models", "endoA_250904_clean.xml")
    expression_rxns_path = os.path.join(project_root, "files", "expressionRxns.csv")

    # Load model and expression data
    model = cobra.io.read_sbml_model(model_path)
    expression_data = pd.read_csv(expression_rxns_path, header=None)

    # --- Model Wrapper ---
    patt = re.compile("__COBAMPGPRDOT__[0-9]{1}")
    replace_alt_transcripts = lambda x: patt.sub("", x)
    model_wrapper = ReconstructionWrapper(
        model=model, ttg_ratio=9999, gpr_gene_parse_function=replace_alt_transcripts
    )

    # iMAT parameters
    exp_thresholds = (0.2, 0.5)  # Low and high expression thresholds
    num_rxns = len(model_wrapper.model_reader.r_ids)
    num_samples = expression_data.shape[1]
    all_solutions = np.zeros((num_rxns, num_samples))

    # Prepare arguments for multiprocessing
    args = [
        (i, expression_data, model_wrapper, exp_thresholds, num_rxns)
        for i in range(num_samples)
    ]

    print(
        f"Using {'multiprocess' if USE_MULTIPROCESS else 'multiprocessing'} with {NUM_CORES} cores"
    )
    print(f"iMAT expression thresholds: {exp_thresholds}")

    overall_start_time = time.time()
    with Pool(processes=NUM_CORES) as pool:
        results = pool.starmap(run_imat_for_sample, args)

    # Separate solutions and solve times
    solutions = []
    solve_times = []
    for result in results:
        if len(result) == 2:  # solution, solve_time
            solutions.append(result[0])
            solve_times.append(result[1])
        else:  # fallback for old format
            solutions.append(result)
            solve_times.append(0.0)

    for sample_idx, solution in enumerate(solutions):
        all_solutions[:, sample_idx] = solution

    overall_time = time.time() - overall_start_time

    # Print timing statistics
    solve_times = np.array(solve_times)
    print(f"\n=== Timing Statistics ===")
    print(f"Total processing time: {overall_time:.2f}s")
    print(f"Average solve time per sample: {np.mean(solve_times):.2f}s")
    print(f"Min solve time: {np.min(solve_times):.2f}s")
    print(f"Max solve time: {np.max(solve_times):.2f}s")
    print(f"Total solve time: {np.sum(solve_times):.2f}s")

    # Save aggregated solutions as CSV
    output_file = os.path.join(project_root, "files", "allsolutions_imat_tutorial.csv")
    pd.DataFrame(all_solutions, index=model_wrapper.model_reader.r_ids).to_csv(
        output_file, header=False
    )
    print(f"\nAggregated iMAT solutions saved to: {output_file}")

    print("\nRun the following script to create reduced models:")
    print("python create_reduced_model_from_imat.py")


if __name__ == "__main__":
    main()
