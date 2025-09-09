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

NUM_CORES = 30


def score_apply(reaction_map_scores):
    return {k: 0 if v is None else v for k, v in reaction_map_scores.items()}


def run_gimme_for_sample(
    sample_idx, expression_data, model_wrapper, idx_objective, num_rxns
):
    print(f"\n=== Running GIMME for sample {sample_idx + 1} ===")
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
    properties = GIMMEProperties(
        exp_vector=[v for k, v in scores.items()],
        obj_frac=0.8,
        objectives=[{idx_objective: 1}],
        preprocess=True,
        flux_threshold=0.8,
        solver="GUROBI",
        reaction_ids=model_wrapper.model_reader.r_ids,
        metabolite_ids=model_wrapper.model_reader.m_ids,
    )
    gimme = GIMME(
        S=model_wrapper.S,
        lb=model_wrapper.lb,
        ub=model_wrapper.ub,
        properties=properties,
    )
    gimme_run = gimme.run()
    flux_solution = gimme.sol.__dict__.get("_Solution__value_map", None)
    # print("GIMME active reaction indices:", gimme_run)
    if flux_solution is not None:
        return [
            flux_solution.get(rxn_id, 0.0)
            for rxn_id in model_wrapper.model_reader.r_ids
        ]
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
    expression_data = pd.read_csv(expression_rxns_path, header=None)

    # --- Model Wrapper ---
    patt = re.compile("__COBAMPGPRDOT__[0-9]{1}")
    replace_alt_transcripts = lambda x: patt.sub("", x)
    model_wrapper = ReconstructionWrapper(
        model=model, ttg_ratio=9999, gpr_gene_parse_function=replace_alt_transcripts
    )

    biomass_rxn_id = "MAR13082"
    idx_objective = model_wrapper.model_reader.r_ids.index(biomass_rxn_id)
    num_rxns = len(model_wrapper.model_reader.r_ids)
    num_samples = expression_data.shape[1]
    all_solutions = np.zeros((num_rxns, num_samples))

    # Prepare arguments for multiprocessing
    args = [
        (i, expression_data, model_wrapper, idx_objective, num_rxns)
        for i in range(num_samples)
    ]

    print(
        f"Using {'multiprocess' if USE_MULTIPROCESS else 'multiprocessing'} with {NUM_CORES} cores"
    )
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
