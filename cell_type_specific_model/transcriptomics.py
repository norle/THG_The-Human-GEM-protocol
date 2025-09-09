# Script for transcriptomics analysis
import numpy as np
import pdb
from cobra.io import read_sbml_model
import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
from model_reduce import model_reduce
from gimme_parallel import gimme_parallel as gimme
import multiprocessing
import copy
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor
import cobra

# Determine the current file's directory and the project root.
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_dir)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
print("Project root directory:", project_root)
# Add the project root to sys.path to access top-level folders like 'functions' and 'models'
if project_root not in sys.path:
    sys.path.append(project_root)


def biomass_fix(model, ratio=0.2):
    """
    Adjusts the biomass reaction lower bound to a specified ratio of the optimal biomass flux.

    This function:
      - Prints the original bounds of the biomass reaction.
      - Optimizes the model (FBA) to determine the optimal biomass flux.
      - Sets the lower bound of the biomass reaction (MAR13082) to the specified ratio of that optimal flux.
      - Re-optimizes the model and prints updated bounds and solution status.

    Parameters:
        model (cobra.Model): The metabolic model.
        ratio (float): The fraction of the optimal biomass flux to set as the new lower bound (default is 0.2).

    Returns:
        cobra.Model: The updated model with the modified biomass reaction lower bound.
    """

    print(
        "Original biomass reaction bounds: ",
        model.reactions.get_by_id("MAR13082").lower_bound,
        model.reactions.get_by_id("MAR13082").upper_bound,
    )

    # Run FBA to get the optimal biomass flux.
    solution = model.optimize()
    optimal_biomass_flux = solution.objective_value
    print("Optimal biomass flux: ", optimal_biomass_flux)
    print("Solution status: ", solution.status)

    # Set a lower bound for the biomass reaction to 80% of the optimal biomass flux
    model.reactions.get_by_id("MAR13082").lower_bound = ratio * optimal_biomass_flux

    # Re-optimize the model after changing the biomass bounds.
    solution = model.optimize()
    optimal_biomass_flux = solution.objective_value
    print("After changing bounds: \nOptimal biomass flux: ", optimal_biomass_flux)
    print("Solution status: ", solution.status)
    print(
        "Lower bound for biomass reaction: ",
        model.reactions.get_by_id("MAR13082").lower_bound,
    )
    print(
        "Final biomass reaction bounds: ",
        model.reactions.get_by_id("MAR13082").lower_bound,
        model.reactions.get_by_id("MAR13082").upper_bound,
    )

    return model


def sgpr_rules(model):
    """
    Extracts sGPR rules from reaction annotations in the model.

    For each reaction in the model, this function searches through its annotations
    for a key containing 'sGPR' and extracts the rule (assumed to be the last component
    of a slash-separated string).

    Parameters:
        model (cobra.Model): The metabolic model.

    Returns:
        list: A list of sGPR rules corresponding to the reactions in the model.
              If a reaction does not have an sGPR rule, its corresponding entry is None.
    """
    sgpr_rules = [None] * len(model.reactions)

    # Extract sGPR rules from the annotations of reactions
    for i, reaction in enumerate(model.reactions):
        if hasattr(reaction, "annotation") and reaction.annotation:
            for key, resources in reaction.annotation.items():
                if "sGPR" in key:
                    sgpr_rule = resources.split("/")[-1]
                    sgpr_rules[i] = sgpr_rule

    return sgpr_rules


def ensemble(model):
    """
    Extracts Ensembl IDs from the gene identifiers in the model.

    For each gene in the model, searches for an Ensembl ID using a regex pattern.
    If an Ensembl ID is found, it is added to the returned list; otherwise, an empty
    string is added.

    Parameters:
        model (cobra.Model): The metabolic model.

    Returns:
        list: A list of Ensembl IDs (or empty strings) corresponding to the model's genes.
    """
    ensembl_model = []
    pattern = r"ENSG\d+"

    # Extract Ensembl IDs from gene identifiers in the model
    for gene in model.genes:
        match = re.search(pattern, gene.id)
        if match:
            ensembl_model.append(match.group(0))
        else:
            ensembl_model.append("")

    return ensembl_model


def index_to_ensembl(model, ensembl_model):
    """
    Updates the gene-protein-reaction (GPR) rules in the model by replacing index tokens
    with corresponding Ensembl IDs.

    The function searches for tokens in the form x(index) in each reaction's GPR rule,
    retrieves the corresponding Ensembl ID from the provided list, and replaces the token.

    Parameters:
        model (cobra.Model): The metabolic model.
        ensembl_model (list): A list of Ensembl IDs corresponding to gene indices.

    Returns:
        list: A list of updated GPR rules with indices replaced by Ensembl IDs.
    """

    new_rules3 = [None] * len(model.reactions)

    # Update GPR rules to replace indices with corresponding Ensembl IDs
    for j, reaction in enumerate(model.reactions):
        rule = reaction.gene_reaction_rule
        tokens = re.findall(r"x\((\d+)\)", rule)
        new_rule = rule

        for token in tokens:
            index = int(token)
            ensg_id = ensembl_model[index] if index < len(ensembl_model) else ""
            new_rule = new_rule.replace(f"x({token})", f"({ensg_id})")

        # Clean up the GPR
        new_rule = new_rule.strip()

        new_rules3[j] = new_rule

    return new_rules3


def extract_pairs(model_path):
    """
    Parses an XML file (model) to extract pairs of HGNC gene symbols and Ensembl IDs.

    The function uses ElementTree to navigate through the XML structure, extracting
    the HGNC symbol and Ensembl ID from specific resource links in the file. In cases
    where duplicate symbols are encountered with different Ensembl IDs, it prints a warning.

    Parameters:
        model_path (str): Path to the XML file containing model information.

    Returns:
        dict: A dictionary mapping HGNC symbols to Ensembl IDs. If duplicates occur,
              the value may be a list of Ensembl IDs.
    """
    tree = ET.parse(model_path)
    root = tree.getroot()
    symbol_and_ensebl = {}

    # Extract pairs of HGNC symbols and Ensembl IDs from an XML file

    # Iterate over all Description elements in the XML.

    for description in root.findall(
        ".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"
    ):
        hgnc_symbol = ""
        ensg = ""

        # Iterate over list items in the Description.
        for li in description.findall(
            ".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li"
        ):
            resource = li.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")

            if resource:
                if "hgnc.symbol/" in resource or "hgcn.symbol/" in resource:
                    hgnc_symbol = resource.split("/")[-1]
                elif "ensembl/ENSG" in resource:
                    ensg = resource.split("/")[-1]

        # If both HGNC and Ensembl ID are found, add them to the dictionary.
        if hgnc_symbol and ensg:
            if hgnc_symbol in symbol_and_ensebl:
                # If duplicate symbol, check whether it is the same Ensembl ID.
                if symbol_and_ensebl[hgnc_symbol] == ensg:
                    continue
                # Convert to list if needed and add the new Ensembl ID.
                if not isinstance(symbol_and_ensebl[hgnc_symbol], list):
                    symbol_and_ensebl[hgnc_symbol] = [symbol_and_ensebl[hgnc_symbol]]
                symbol_and_ensebl[hgnc_symbol].append(ensg)
            else:
                symbol_and_ensebl[hgnc_symbol] = ensg

    return symbol_and_ensebl


def sgpr_to_ensembl(model, symbol_and_ensebl):
    """
    Converts GPR rules in the model from gene symbols/indices to Ensembl IDs using
    the provided mapping.

    For each reaction, the function scans the gene reaction rule for gene tokens
    (ignoring logical operators). It then replaces these tokens with the corresponding
    Ensembl IDs from the symbol_and_ensebl dictionary. Unmatched genes are reported.

    Parameters:
        model (cobra.Model): The metabolic model.
        symbol_and_ensebl (dict): Dictionary mapping HGNC symbols to Ensembl IDs.

    Returns:
        list: A list of updated GPR rules with gene tokens replaced by Ensembl IDs.
    """

    new_rules3 = [None] * len(model.reactions)
    unmatched_indices = []  # Gene tokens not found in the mapping.
    unmatched_rules = []  # GPR rules with missing Ensembl IDs.
    unmatched_row_numbers = []  # Row numbers of reactions with unmatched genes.

    for j, reaction in enumerate(model.reactions):
        rule = reaction.gene_reaction_rule
        new_rule = rule

        # Find all words (potential gene tokens) in the rule.
        genes = re.findall(r"\w+", rule)
        # Remove logical operator tokens.
        genes = [gene for gene in genes if gene not in ["or", "and"]]
        for gene in genes:
            # Check if gene is already an Ensembl ID or is present in the mapping.
            if gene in symbol_and_ensebl.values():
                ensg_id = gene
            elif gene.startswith("ENSG"):
                ensg_id = gene  # Already in Ensembl format.
            elif gene in symbol_and_ensebl.keys():
                # If multiple Ensembl IDs exist, choose the first.
                if isinstance(symbol_and_ensebl[gene], list):
                    print(
                        f"Gene {gene} has multiple Ensembl IDs: {symbol_and_ensebl[gene]}"
                    )
                    ensg_id = symbol_and_ensebl[gene][0]
                else:
                    ensg_id = symbol_and_ensebl[gene]
            else:
                ensg_id = ""
                print(f"Gene {gene} not found in the Ensembl model")
                unmatched_indices.append(gene)
                unmatched_rules.append(rule)
                unmatched_row_numbers.append(j)
            # Replace the gene token in the rule with the Ensembl ID.
            new_rule = re.sub(r"\b{}\b".format(re.escape(gene)), ensg_id, new_rule)

        new_rule = new_rule.strip()
        new_rules3[j] = new_rule

    return new_rules3


class CustomExpression:
    """
    Implements a custom expression object that allows arithmetic operations

    The class defines custom behavior for logical AND (&) and OR (|) operators.
    """

    def __init__(self, value):
        """
        Initialize with a numeric expression value.

        Parameters:
            value (float): The expression value.
        """
        self.value = value

    def __and__(self, other):
        """
        Implements the AND logic.

        Returns:
            CustomExpression: For AND, if either value is 0, returns the maximum;
                              otherwise, returns the minimum.
        """
        if self.value == 0 or other.value == 0:
            return CustomExpression(max(self.value, other.value))
        else:
            return CustomExpression(min(self.value, other.value))

    def __or__(self, other):
        """
        Implements the OR logic.

        Returns:
            CustomExpression: The maximum of the two values.
        """
        return CustomExpression(max(self.value, other.value))

    def __repr__(self):
        """Make the object printable for debugging."""
        return str(self.value)


def preprocess_rule(rule, ensembl_map):
    """
    Preprocesses a gene reaction rule to standardize Ensembl IDs and embed expression values.

    The function performs the following steps:
      - Finds all Ensembl IDs in the rule that match the pattern ENSG followed by 11 digits
        and optional extra characters, truncating any extra characters.
      - Replaces these Ensembl IDs with a call to CustomExpression using the corresponding
        expression value from ensembl_map.
      - Converts logical operator tokens ('[', ']', 'or', 'and') into standard symbols.

    Parameters:
        rule (str): The gene reaction rule as a string.
        ensembl_map (dict): Dictionary mapping Ensembl IDs to expression values.

    Returns:
        str: The preprocessed rule ready for evaluation.
    """

    # Find all Ensembl IDs and remove any extra characters beyond the standard 15 characters (ENSG + 11 digits)
    matches = re.findall(r"ENSG\d{11}[A-Za-z0-9]*", rule)
    # Process the matches by length
    matches.sort(key=len, reverse=True)
    for match in matches:
        if len(match) > 15:
            new_match = match[:15]
            rule = rule.replace(match, new_match)

    # Replace gene IDs with their corresponding expression values
    matches = re.findall(r"ENSG\d+", rule)
    matches.sort(key=len, reverse=True)
    for match in matches:
        new_match = match[:15]
        rule = rule.replace(match, new_match)
        expression_value = ensembl_map.get(new_match, -1)  # Default to -1 if not found
        rule = rule.replace(new_match, f"CustomExpression({expression_value})")

    # Perform regex replacements for logical operators
    rule = re.sub(r"\[", "(", rule)
    rule = re.sub(r"\]", ")", rule)
    rule = re.sub(r"\bor\b", "|", rule)
    rule = re.sub(r"\band\b", "&", rule)

    return rule


def process_expression_data(new_rules3, lung_data):
    """
    Processes gene expression data for multiple samples and evaluates the gene reaction rules.
    Optimized version with pre-compiled patterns and vectorized operations.
    """
    # Load the expression data in GCT format: skip first two lines, use tab separator
    lung_data_df = pd.read_csv(lung_data, sep="\t", skiprows=2)
    num_samples = lung_data_df.shape[1] - 2  # Exclude Name and Description columns

    # Pre-extract all gene IDs and convert to numpy array for faster access
    gene_ids = lung_data_df.iloc[:, 0].astype(str).str.split(".").str[0].values
    expression_data = lung_data_df.iloc[:, 2:].values  # All sample expression data

    # Pre-compile regex patterns and extract all genes from rules once
    ensembl_pattern = re.compile(r"ENSG\d{11}")

    # Extract all unique genes from rules and their positions
    rule_genes_map = {}  # Maps rule index to list of gene IDs in that rule
    all_rule_genes = set()

    for i, rule in enumerate(new_rules3):
        if isinstance(rule, str):
            genes_in_rule = ensembl_pattern.findall(rule)
            rule_genes_map[i] = genes_in_rule
            all_rule_genes.update(genes_in_rule)
        else:
            rule_genes_map[i] = []

    # Create mapping from gene_id to index in expression data
    gene_to_idx = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}

    # Pre-filter to only genes that exist in expression data
    existing_genes = all_rule_genes.intersection(set(gene_ids))
    missing_genes = all_rule_genes - existing_genes

    print(f"Performance stats:")
    print(f"Total unique genes in model rules: {len(all_rule_genes)}")
    print(f"Total genes in GCT file: {len(gene_ids)}")
    print(f"Found (overlap) genes: {len(existing_genes)}")
    print(f"Missing genes from GCT: {len(missing_genes)}")
    print(f"Coverage: {len(existing_genes)/len(all_rule_genes)*100:.1f}%")

    # Pre-compile rule templates for faster processing
    compiled_rules = []
    for i, rule in enumerate(new_rules3):
        if isinstance(rule, str) and rule_genes_map[i]:
            # Pre-process rule structure
            processed_rule = rule
            # Clean up logical operators
            processed_rule = re.sub(r"\[", "(", processed_rule)
            processed_rule = re.sub(r"\]", ")", processed_rule)
            processed_rule = re.sub(r"\bor\b", "|", processed_rule)
            processed_rule = re.sub(r"\band\b", "&", processed_rule)
            compiled_rules.append((i, processed_rule, rule_genes_map[i]))
        else:
            compiled_rules.append((i, None, []))

    def process_sample_batch(sample_indices):
        """Process a batch of samples"""
        batch_results = {}

        for sample_idx in sample_indices:
            sample_col = sample_idx - 2  # Convert to 0-based sample index

            # Vectorized lookup of expression values for this sample
            sample_expressions = expression_data[:, sample_col]

            # Create expression map for existing genes only
            ensembl_map = {}
            for gene_id, expr_idx in gene_to_idx.items():
                if gene_id in existing_genes:
                    ensembl_map[gene_id] = sample_expressions[expr_idx]

            # Process rules for this sample
            sample_rules = [None] * len(new_rules3)

            for rule_idx, processed_rule, genes_in_rule in compiled_rules:
                if processed_rule is None:
                    sample_rules[rule_idx] = None
                    continue

                # Replace genes with expression values
                final_rule = processed_rule
                for gene in genes_in_rule:
                    expr_val = ensembl_map.get(gene, -1)
                    final_rule = final_rule.replace(
                        gene, f"CustomExpression({expr_val})"
                    )

                sample_rules[rule_idx] = final_rule

            batch_results[sample_col] = sample_rules

        return batch_results

    # Process samples in parallel batches
    batch_size = max(1, num_samples // (multiprocessing.cpu_count() * 2))
    sample_indices = list(range(2, lung_data_df.shape[1]))

    rules_samples = [[None] * num_samples for _ in range(len(new_rules3))]

    # Use ThreadPoolExecutor for I/O bound operations
    with ThreadPoolExecutor(
        max_workers=min(8, multiprocessing.cpu_count())
    ) as executor:
        # Split samples into batches
        batches = [
            sample_indices[i : i + batch_size]
            for i in range(0, len(sample_indices), batch_size)
        ]

        # Process batches in parallel
        futures = [executor.submit(process_sample_batch, batch) for batch in batches]

        # Collect results with progress bar
        for future in tqdm(futures, desc="Processing sample batches"):
            batch_results = future.result()

            # Merge batch results
            for sample_col, sample_rules in batch_results.items():
                for rule_idx in range(len(new_rules3)):
                    rules_samples[rule_idx][sample_col] = sample_rules[rule_idx]

    # Evaluate the preprocessed rules using vectorized operations
    print("Evaluating gene expression levels using gene rules...")
    expressionRxns = np.full((len(rules_samples), num_samples), -1.0, dtype=np.float32)

    # Create a restricted evaluation environment
    local_env = {"CustomExpression": CustomExpression}

    # Vectorized evaluation
    def evaluate_sample_batch(sample_indices):
        """Evaluate rules for a batch of samples"""
        batch_scores = {}

        for sample_idx in sample_indices:
            rules3 = [rules_samples[i][sample_idx] for i in range(len(rules_samples))]
            frxnscores = np.full(len(rules3), -1.0, dtype=np.float32)

            for k, rule in enumerate(rules3):
                if rule and rule.strip():
                    try:
                        frxnscores[k] = eval(
                            rule, {"__builtins__": {}}, local_env
                        ).value
                    except:
                        frxnscores[k] = -1.0

            batch_scores[sample_idx] = frxnscores

        return batch_scores

    # Evaluate in parallel batches
    sample_indices = list(range(num_samples))
    eval_batch_size = max(1, num_samples // multiprocessing.cpu_count())

    with ThreadPoolExecutor(
        max_workers=min(8, multiprocessing.cpu_count())
    ) as executor:
        eval_batches = [
            sample_indices[i : i + eval_batch_size]
            for i in range(0, len(sample_indices), eval_batch_size)
        ]

        eval_futures = [
            executor.submit(evaluate_sample_batch, batch) for batch in eval_batches
        ]

        for future in tqdm(eval_futures, desc="Evaluating expression rules"):
            batch_scores = future.result()

            for sample_idx, scores in batch_scores.items():
                expressionRxns[:, sample_idx] = scores

    print("Finished processing expression data.")
    return rules_samples, expressionRxns


# Main function


def main():
    """
    Main function to execute the transcriptomics analysis pipeline.

    Workflow:
      - Defines file paths for the model and expression data.
      - Loads the metabolic model (SBML format) and applies biomass_fix.
      - Extracts gene symbol and Ensembl ID pairs from the model XML.
      - Converts sGPR rules to use Ensembl IDs.
      - Processes the expression data (preprocess and evaluate gene rules).
      - Saves the processed expression values to a CSV file.
      - Reads exchange reaction IDs relevant in the specific cell type.
      - Runs the parallelized GIMME algorithm to obtain flux solutions.
      - Calls model_reduce to generate a tailored model based on the flux solutions.
      - Prints the output location of the tailored model.
    """
    # Define paths using os.path.join for cross-platform compatibility
    cobra.Configuration().solver = "gurobi"
    model_path = os.path.join(
        project_root, "models", "endoA_250904_clean.xml"
    )  # Output model after adding transport reactions
    lung_data = os.path.join(
        project_root, "files", "gene_reads_v10_lung.gct"
    )  # GTEx expression data
    expression_rxns_path = os.path.join(project_root, "files", "expressionRxns.csv")

    # Load the model
    model = read_sbml_model(model_path)

    if os.path.exists(expression_rxns_path):
        print(
            f"ExpressionRxns file already exists at {expression_rxns_path}. Loading data..."
        )
        expressionRxns = pd.read_csv(expression_rxns_path, header=None).values
    else:
        model = biomass_fix(model)
        # Extract gene symbol and Ensembl ID pairs from the model XML.
        symbol_and_ensembl = extract_pairs(model_path)
        # Convert sGPR rules to use Ensembl IDs.
        new_rules2 = sgpr_to_ensembl(model, symbol_and_ensembl)
        # Process the expression data to generate evaluated expression values.
        rules_samples, expressionRxns = process_expression_data(new_rules2, lung_data)

        # Convert rules_samples to a more memory-efficient data type
        rules_samples = np.array(
            rules_samples, dtype=object
        )  # Use dtype=object for mixed types
        expressionRxns = np.array(
            expressionRxns, dtype=np.float32
        )  # Use float32 for smaller memory footprint

        expressionRxns_df = pd.DataFrame(expressionRxns)
        expressionRxns_df.to_csv(expression_rxns_path, index=False, header=False)

    # Exchange reactions common between the general model and the cell type specific model (obtained from match_exch_rxns script)
    exchange_rs = [rxn.id for rxn in model.reactions if rxn.boundary]

    # Create a copy for flux computations so that the original model remains unmodified
    model_flux = copy.deepcopy(model)
    all_solutions = gimme(model_flux, expressionRxns, exchange_rs, num_workers=4)

    # Tailor the model based on the flux solutions from the reconstruction algorithm
    output_path = os.path.join(
        project_root, "models", "endoA_290905.xml"
    )  # Output path for the tailored model

    model_reduce(model, all_solutions, output_path)

    print("Model tailored and saved to ", output_path)


if __name__ == "__main__":
    main()
