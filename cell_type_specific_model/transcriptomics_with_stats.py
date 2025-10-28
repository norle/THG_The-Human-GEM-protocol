# Script for transcriptomics analysis with reaction statistics
import numpy as np
import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
from cobra.io import read_sbml_model
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
project_root = os.path.abspath(os.path.join(current_dir, ".."))
print("Project root directory:", project_root)
if project_root not in sys.path:
    sys.path.append(project_root)


def biomass_fix(model, ratio=0.2):
    print(
        "Original biomass reaction bounds: ",
        model.reactions.get_by_id("MAR13082").lower_bound,
        model.reactions.get_by_id("MAR13082").upper_bound,
    )
    solution = model.optimize()
    optimal_biomass_flux = solution.objective_value
    print("Optimal biomass flux: ", optimal_biomass_flux)
    print("Solution status: ", solution.status)
    model.reactions.get_by_id("MAR13082").lower_bound = ratio * optimal_biomass_flux
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


def extract_pairs(model_path):
    tree = ET.parse(model_path)
    root = tree.getroot()
    symbol_and_ensebl = {}
    for description in root.findall(
        ".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"
    ):
        hgnc_symbol = ""
        ensg = ""
        for li in description.findall(
            ".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li"
        ):
            resource = li.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
            if resource:
                if "hgnc.symbol/" in resource or "hgcn.symbol/" in resource:
                    hgnc_symbol = resource.split("/")[-1]
                elif "ensembl/ENSG" in resource:
                    ensg = resource.split("/")[-1]
        if hgnc_symbol and ensg:
            if hgnc_symbol in symbol_and_ensebl:
                if symbol_and_ensebl[hgnc_symbol] == ensg:
                    continue
                if not isinstance(symbol_and_ensebl[hgnc_symbol], list):
                    symbol_and_ensebl[hgnc_symbol] = [symbol_and_ensebl[hgnc_symbol]]
                symbol_and_ensebl[hgnc_symbol].append(ensg)
            else:
                symbol_and_ensebl[hgnc_symbol] = ensg
    return symbol_and_ensebl


def sgpr_to_ensembl(model, symbol_and_ensebl):
    new_rules3 = [None] * len(model.reactions)
    for j, reaction in enumerate(model.reactions):
        rule = reaction.gene_reaction_rule
        new_rule = rule
        genes = re.findall(r"\w+", rule)
        genes = [gene for gene in genes if gene not in ["or", "and"]]
        for gene in genes:
            if gene in symbol_and_ensebl.values():
                ensg_id = gene
            elif gene.startswith("ENSG"):
                ensg_id = gene
            elif gene in symbol_and_ensebl.keys():
                if isinstance(symbol_and_ensebl[gene], list):
                    ensg_id = symbol_and_ensebl[gene][0]
                else:
                    ensg_id = symbol_and_ensebl[gene]
            else:
                ensg_id = ""
            new_rule = re.sub(r"\b{}\b".format(re.escape(gene)), ensg_id, new_rule)
        new_rule = new_rule.strip()
        new_rules3[j] = new_rule
    return new_rules3


class CustomExpression:
    def __init__(self, value):
        self.value = value

    def __and__(self, other):
        if self.value == 0 or other.value == 0:
            return CustomExpression(max(self.value, other.value))
        else:
            return CustomExpression(min(self.value, other.value))

    def __or__(self, other):
        return CustomExpression(max(self.value, other.value))

    def __repr__(self):
        return str(self.value)


def process_expression_data(new_rules3, lung_data):
    lung_data_df = pd.read_csv(lung_data, sep="\t", skiprows=2)
    num_samples = lung_data_df.shape[1] - 2
    gene_ids = lung_data_df.iloc[:, 0].astype(str).str.split(".").str[0].values
    expression_data = lung_data_df.iloc[:, 2:].values
    ensembl_pattern = re.compile(r"ENSG\d{11}")
    rule_genes_map = {}
    all_rule_genes = set()
    for i, rule in enumerate(new_rules3):
        if isinstance(rule, str):
            genes_in_rule = ensembl_pattern.findall(rule)
            rule_genes_map[i] = genes_in_rule
            all_rule_genes.update(genes_in_rule)
        else:
            rule_genes_map[i] = []
    gene_to_idx = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}
    existing_genes = all_rule_genes.intersection(set(gene_ids))
    compiled_rules = []
    for i, rule in enumerate(new_rules3):
        if isinstance(rule, str) and rule_genes_map[i]:
            processed_rule = rule
            processed_rule = re.sub(r"\[", "(", processed_rule)
            processed_rule = re.sub(r"\]", ")", processed_rule)
            processed_rule = re.sub(r"\bor\b", "|", processed_rule)
            processed_rule = re.sub(r"\band\b", "&", processed_rule)
            compiled_rules.append((i, processed_rule, rule_genes_map[i]))
        else:
            compiled_rules.append((i, None, []))

    def process_sample_batch(sample_indices):
        batch_results = {}
        for sample_idx in sample_indices:
            sample_col = sample_idx - 2
            sample_expressions = expression_data[:, sample_col]
            ensembl_map = {}
            for gene_id, expr_idx in gene_to_idx.items():
                if gene_id in existing_genes:
                    ensembl_map[gene_id] = sample_expressions[expr_idx]
            sample_rules = [None] * len(new_rules3)
            for rule_idx, processed_rule, genes_in_rule in compiled_rules:
                if processed_rule is None:
                    sample_rules[rule_idx] = None
                    continue
                final_rule = processed_rule
                for gene in genes_in_rule:
                    expr_val = ensembl_map.get(gene, -1)
                    final_rule = final_rule.replace(
                        gene, f"CustomExpression({expr_val})"
                    )
                sample_rules[rule_idx] = final_rule
            batch_results[sample_col] = sample_rules
        return batch_results

    batch_size = max(1, num_samples // (multiprocessing.cpu_count() * 2))
    sample_indices = list(range(2, lung_data_df.shape[1]))
    rules_samples = [[None] * num_samples for _ in range(len(new_rules3))]
    with ThreadPoolExecutor(
        max_workers=min(8, multiprocessing.cpu_count())
    ) as executor:
        batches = [
            sample_indices[i : i + batch_size]
            for i in range(0, len(sample_indices), batch_size)
        ]
        futures = [executor.submit(process_sample_batch, batch) for batch in batches]
        for future in tqdm(futures, desc="Processing sample batches"):
            batch_results = future.result()
            for sample_col, sample_rules in batch_results.items():
                for rule_idx in range(len(new_rules3)):
                    rules_samples[rule_idx][sample_col] = sample_rules[rule_idx]
    print("Evaluating gene expression levels using gene rules...")
    expressionRxns = np.full((len(rules_samples), num_samples), -1.0, dtype=np.float32)
    local_env = {"CustomExpression": CustomExpression}

    def evaluate_sample_batch(sample_indices):
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


def plot_hybrid_hist(values, outpath, title="Histogram", xlabel="Value"):
    """Plot a hybrid histogram: single bin for <=0 and log-spaced bins for positives.

    Saves the figure to outpath.
    """
    import matplotlib.pyplot as plt

    values = np.asarray(values)
    values = values[~np.isnan(values)]
    plt.figure(figsize=(8, 6))

    neg_mask = values <= 0
    pos_vals = values[values > 0]
    neg_count = int(np.sum(neg_mask))

    if pos_vals.size > 0:
        min_pos = pos_vals.min()
        max_pos = pos_vals.max()
        if min_pos == max_pos:
            min_pos = min_pos / 10.0 if min_pos > 0 else 1e-3
            max_pos = max_pos * 10.0 if max_pos > 0 else 1.0
        pos_bins = np.logspace(np.log10(min_pos), np.log10(max_pos), 50)
        pos_counts, pos_edges = np.histogram(pos_vals, bins=pos_bins)

        if neg_count > 0:
            neg_bar_x = pos_edges[0] / (10**0.5)
            neg_bar_width = pos_edges[0] / (10**0.25)
            plt.bar(
                neg_bar_x,
                neg_count,
                width=neg_bar_width,
                align="center",
                alpha=0.7,
                label="<= 0",
            )

        widths = pos_edges[1:] - pos_edges[:-1]
        plt.bar(
            pos_edges[:-1],
            pos_counts,
            width=widths,
            align="edge",
            alpha=0.7,
            label="> 0",
        )

        plt.xscale("log")
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.title(title)
        if neg_count > 0:
            current_ticks = plt.xticks()[0]
            new_ticks = np.unique(np.concatenate(([neg_bar_x], current_ticks)))
            labels = []
            for t in new_ticks:
                if np.isclose(t, neg_bar_x):
                    labels.append("<=0")
                else:
                    labels.append("{:.2g}".format(t))
            plt.xticks(new_ticks, labels, rotation=45)
    else:
        # all non-positive
        plt.bar(0, neg_count, width=1, align="center", alpha=0.7)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.title(title)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def analyze_models(large_model_path, subset_model_path, lung_data, output_prefix=None):
    """Compare two models (subset and larger), compute expression table for the larger model,
    flag reactions missing in the subset, and produce two histograms:
      - all reaction means (larger model)
      - means of reactions missing in the subset

    Parameters:
      large_model_path: path to SBML of the larger model
      subset_model_path: path to SBML of the subset model
      lung_data: path to GCT expression file
      output_prefix: prefix for output files (optional)
    """
    # Load models
    large_model = read_sbml_model(large_model_path)
    subset_model = read_sbml_model(subset_model_path)

    # mapping from model annotations
    symbol_and_ensebl = extract_pairs(large_model_path)

    # convert rules to ensembl for the large model
    new_rules_large = sgpr_to_ensembl(large_model, symbol_and_ensebl)
    rules_samples, expressionRxns = process_expression_data(new_rules_large, lung_data)
    expressionRxns = np.array(expressionRxns, dtype=np.float32)

    # Build DataFrame
    reaction_ids = [rxn.id for rxn in large_model.reactions]
    df = pd.DataFrame(expressionRxns)
    df.insert(0, "reaction_id", reaction_ids)
    df.index.name = "index"
    df["mean"] = df.iloc[:, 1:].mean(axis=1)
    df["std"] = df.iloc[:, 1:].std(axis=1)

    # Flag missing reactions
    subset_ids = set([rxn.id for rxn in subset_model.reactions])
    df["missing_in_subset"] = ~df["reaction_id"].isin(subset_ids)

    # Prepare outputs
    out_prefix = (
        output_prefix or os.path.splitext(os.path.basename(large_model_path))[0]
    )
    excel_path = os.path.join(
        project_root, f"files/{out_prefix}_expression_with_missing.xlsx"
    )
    df.to_excel(excel_path, index=True)

    # Histograms
    all_hist_path = os.path.join(project_root, f"files/{out_prefix}_all_means_hist.png")
    missing_hist_path = os.path.join(
        project_root, f"files/{out_prefix}_missing_means_hist.png"
    )

    plot_hybrid_hist(
        df["mean"].values,
        all_hist_path,
        title=f"All reaction means: {out_prefix}",
        xlabel="Mean expression",
    )
    plot_hybrid_hist(
        df.loc[df["missing_in_subset"], "mean"].values,
        missing_hist_path,
        title=f"Missing reaction means: {out_prefix}",
        xlabel="Mean expression",
    )

    print(f"Wrote expression table with missing flag to {excel_path}")
    print(f"Saved histograms: {all_hist_path} and {missing_hist_path}")
    return df


def main():
    cobra.Configuration().solver = "gurobi"
    model_path = os.path.join(project_root, "models", "endoA_250904_clean.xml")
    lung_data = os.path.join(project_root, "files", "gene_reads_v10_lung.gct")
    output_excel_path = os.path.join(
        project_root, "files", "expressionRxns_with_stats.xlsx"
    )
    # Load the model
    model = read_sbml_model(model_path)
    model = biomass_fix(model)
    symbol_and_ensebl = extract_pairs(model_path)
    new_rules2 = sgpr_to_ensembl(model, symbol_and_ensebl)
    rules_samples, expressionRxns = process_expression_data(new_rules2, lung_data)
    expressionRxns = np.array(expressionRxns, dtype=np.float32)
    # Prepare DataFrame with reaction index and id
    reaction_ids = [rxn.id for rxn in model.reactions]
    df = pd.DataFrame(expressionRxns)
    df.insert(0, "reaction_id", reaction_ids)
    df.index.name = "index"
    # Calculate mean and std for each reaction
    df["mean"] = df.iloc[:, 1:].mean(axis=1)
    df["std"] = df.iloc[:, 1:].std(axis=1)

    # Export to Excel
    # df.to_excel(output_excel_path, index=True)
    # print(f"ExpressionRxns with stats exported to {output_excel_path}")

    # Plot and save histogram of means using a hybrid approach:
    # - a single bin for values <= 0 (negatives / zero)
    # - log-spaced bins for positive values (displayed on a log x-axis)
    # - log y-axis for counts (preserves original behaviour of log=True)
    import matplotlib.pyplot as plt

    values = df["mean"].dropna().values
    plt.figure(figsize=(8, 6))

    # Separate non-positive and positive values
    neg_mask = values <= 0
    pos_mask = values > 0
    neg_count = int(np.sum(neg_mask))
    pos_vals = values[pos_mask]

    if pos_vals.size > 0:
        # Prepare log-spaced bins for positive values
        min_pos = pos_vals.min()
        max_pos = pos_vals.max()
        # if all positives are equal, expand range a bit to make bins
        if min_pos == max_pos:
            min_pos = min_pos / 10.0 if min_pos > 0 else 1e-3
            max_pos = max_pos * 10.0 if max_pos > 0 else 1.0
        pos_bins = np.logspace(np.log10(min_pos), np.log10(max_pos), 50)
        pos_counts, pos_edges = np.histogram(pos_vals, bins=pos_bins)

        # Plot negative/zero bin as a single bar placed left of the first positive bin
        if neg_count > 0:
            # place the negative bar at a small positive x (left of first positive bin)
            neg_bar_x = pos_edges[0] / (10**0.5)
            neg_bar_width = pos_edges[0] / (10**0.25)
            plt.bar(
                neg_bar_x,
                neg_count,
                width=neg_bar_width,
                align="center",
                alpha=0.7,
                label="<= 0",
            )

        # Plot positive bins using bars aligned to edges
        widths = pos_edges[1:] - pos_edges[:-1]
        plt.bar(
            pos_edges[:-1],
            pos_counts,
            width=widths,
            align="edge",
            alpha=0.7,
            label="> 0",
        )

        # Use log scale for x (positives) and y (counts)
        plt.xscale("log")
        # plt.yscale("log")
        plt.xlabel("Mean expression (positives log-scaled; negatives binned at left)")
        plt.ylabel("Count ")
        plt.title("Histogram of Reaction Means (hybrid bins)")

        # Add a custom tick for the negative bin and label it "<=0"
        if neg_count > 0:
            current_ticks = plt.xticks()[0]
            new_ticks = np.unique(np.concatenate(([neg_bar_x], current_ticks)))
            labels = []
            for t in new_ticks:
                if np.isclose(t, neg_bar_x):
                    labels.append("<=0")
                else:
                    labels.append("{:.2g}".format(t))
            plt.xticks(new_ticks, labels, rotation=45)
    else:
        # No positive values: simple bar for non-positive counts
        plt.bar(0, neg_count, width=1, align="center", alpha=0.7)
        plt.xlabel("Mean expression")
        plt.ylabel("Count")
        plt.title("Histogram of Reaction Means (all non-positive)")

    hist_path = os.path.join(project_root, "files", "expressionRxns_means_hist.png")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()
    print(f"Histogram of means saved to {hist_path}")


if __name__ == "__main__":
    #main()
    large_model_path = "models/endoA_250903_clean.xml"
    subset_model_path = "models/endoA_250904_clean.xml"
    lung_data = "files/gene_reads_v10_lung.gct"
    analyze_models(large_model_path, subset_model_path, lung_data)
