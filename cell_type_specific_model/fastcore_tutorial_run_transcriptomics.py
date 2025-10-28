import pandas as pd
import numpy as np
import cobra
import re
from troppo.omics.readers.generic import TabularReader
from troppo.methods_wrappers import ModelBasedWrapper, ReconstructionWrapper
from troppo.omics.integration import CustomSelectionIntegrationStrategy
from troppo.methods.reconstruction.fastcore import FASTcore, FastcoreProperties
import os
import copy
import time
from math import log

try:
    from multiprocess import Pool

    USE_MULTIPROCESS = True
except ImportError:
    from multiprocessing import Pool

    USE_MULTIPROCESS = False

NUM_CORES = 30


def run_fastcore_for_sample(
    sample_idx, expression_data, model_wrapper, threshold, protected_reactions, num_rxns
):
    start_time = time.time()
    print(f"\n=== Running FastCORE for sample {sample_idx + 1} ===")
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

    # Custom integration function for FastCORE
    def integration_fx(reaction_map_scores):
        return [
            [
                k
                for k, v in reaction_map_scores.get_scores().items()
                if (v is not None and v > threshold) or k in protected_reactions
            ]
        ]

    threshold_integration = CustomSelectionIntegrationStrategy(
        group_functions=[integration_fx]
    )
    threshold_scores = threshold_integration.integrate(data_map=data_map)

    # Get the index of the core reaction set
    ordered_ids = {r: i for i, r in enumerate(model_wrapper.model_reader.r_ids)}
    core_idx = [
        [ordered_ids[k] for k in l if k in ordered_ids] for l in threshold_scores
    ]

    # Flatten core_idx if it's nested
    if core_idx and isinstance(core_idx[0], list):
        core_reactions_flat = core_idx[0]
    else:
        core_reactions_flat = core_idx

    print(
        f"Sample {sample_idx + 1}: {len(core_reactions_flat)} core reactions identified"
    )

    # Define FastCORE properties
    properties = FastcoreProperties(
        core=core_reactions_flat,
        flux_threshold=1e-6,
        solver="GUROBI",  # Changed from CPLEX to GUROBI
    )

    # Run FastCORE algorithm
    fastcore = FASTcore(
        S=model_wrapper.S,
        lb=model_wrapper.lb,
        ub=model_wrapper.ub,
        properties=properties,
    )

    solve_start_time = time.time()
    active_reactions = fastcore.run()
    solve_time = time.time() - solve_start_time
    total_time = time.time() - start_time

    print(
        f"Sample {sample_idx + 1}: {len(active_reactions)} FastCORE reactions | "
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

    # FastCORE parameters
    threshold = 5 * log(2)  # Expression threshold for core reactions
    protected_reactions = ["MAR13082"]  # Biomass reaction - always include

    num_rxns = len(model_wrapper.model_reader.r_ids)
    num_samples = expression_data.shape[1]
    all_solutions = np.zeros((num_rxns, num_samples))

    # Prepare arguments for multiprocessing
    args = [
        (i, expression_data, model_wrapper, threshold, protected_reactions, num_rxns)
        for i in range(num_samples)
    ]

    print(
        f"Using {'multiprocess' if USE_MULTIPROCESS else 'multiprocessing'} with {NUM_CORES} cores"
    )
    print(f"FastCORE expression threshold: {threshold:.3f}")
    print(f"Protected reactions: {protected_reactions}")

    overall_start_time = time.time()
    with Pool(processes=NUM_CORES) as pool:
        results = pool.starmap(run_fastcore_for_sample, args)

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
    output_file = os.path.join(
        project_root, "files", "allsolutions_fastcore_tutorial.csv"
    )
    pd.DataFrame(all_solutions, index=model_wrapper.model_reader.r_ids).to_csv(
        output_file, header=False
    )
    print(f"\nAggregated FastCORE solutions saved to: {output_file}")

    print("\nRun the following script to create reduced models:")
    print("python create_reduced_model_from_fastcore.py")


if __name__ == "__main__":
    main()
