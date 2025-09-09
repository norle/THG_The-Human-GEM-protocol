import pandas as pd
import numpy as np
import cobra
import re
from troppo.omics.readers.generic import TabularReader
from troppo.methods_wrappers import ModelBasedWrapper, ReconstructionWrapper
from troppo.omics.integration import ContinuousScoreIntegrationStrategy
from troppo.methods.reconstruction.gimme import GIMME, GIMMEProperties


def main():
    # --- Initial Setup ---
    patt = re.compile("__COBAMPGPRDOT__[0-9]{1}")
    replace_alt_transcripts = lambda x: patt.sub("", x)

    # Load model and expression data
    model = cobra.io.read_sbml_model("data/HumanGEM_Consistent.xml")
    expression_data = pd.read_csv("data/expression_data.csv", index_col=0)

    # --- OmicsContainer ---
    omics_container = TabularReader(
        path_or_df=expression_data,
        nomenclature="entrez_id",
        omics_type="transcriptomics",
    ).to_containers()
    single_sample = omics_container[0]

    # --- Model Wrapper ---
    model_wrapper = ReconstructionWrapper(
        model=model, ttg_ratio=9999, gpr_gene_parse_function=replace_alt_transcripts
    )

    # --- Map gene IDs in the data to model IDs ---
    data_map = single_sample.get_integrated_data_map(
        model_reader=model_wrapper.model_reader, and_func=min, or_func=sum
    )

    # --- Integrate Scores ---
    def score_apply(reaction_map_scores):
        return {k: 0 if v is None else v for k, v in reaction_map_scores.items()}

    continuous_integration = ContinuousScoreIntegrationStrategy(score_apply=score_apply)
    scores = continuous_integration.integrate(data_map=data_map)

    # --- Run GIMME ---
    idx_objective = model_wrapper.model_reader.r_ids.index("biomass_human")
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

    # Access flux solution
    flux_solution = gimme.sol.__dict__.get("_Solution__value_map", None)
    print("GIMME active reaction indices:", gimme_run)
    print("GIMME flux solution:", flux_solution)


if __name__ == "__main__":
    main()
