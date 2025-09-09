import pandas as pd
import numpy as np
import cobra
import os
import matplotlib.pyplot as plt


def create_reduced_model_from_imat_csv(
    csv_file_path, original_model_path, output_model_path, presence_threshold=0.1
):
    """
    Create a reduced model based on reaction presence in iMAT solutions.

    Parameters:
    - csv_file_path: Path to the CSV file with iMAT solutions (rows=reactions, cols=samples)
    - original_model_path: Path to the original SBML model
    - output_model_path: Path to save the reduced model
    - presence_threshold: Minimum fraction of samples where reaction must be active (default 0.1 = 10%)
    """

    # Load the iMAT solutions CSV (with reaction IDs as index)
    print("Loading iMAT solutions...")
    imat_df = pd.read_csv(csv_file_path, index_col=0, header=None)
    print(
        f"Loaded iMAT data: {imat_df.shape[0]} reactions × {imat_df.shape[1]} samples"
    )

    # Convert to numpy array for calculations
    imat_matrix = imat_df.values

    # iMAT solutions are already binary (0 or 1), so no need to discretize
    # Calculate mean presence of each reaction across samples
    mean_presence = np.mean(imat_matrix, axis=1)

    # Get reaction IDs that meet the presence threshold
    active_reactions = imat_df.index[mean_presence >= presence_threshold].tolist()

    print(
        f"Reactions meeting {presence_threshold*100}% presence threshold: {len(active_reactions)}/{len(imat_df.index)}"
    )

    # Load original model
    print("Loading original model...")
    model = cobra.io.read_sbml_model(original_model_path)
    original_reactions = len(model.reactions)

    # Create reduced model by removing inactive reactions
    reactions_to_remove = []
    for reaction in model.reactions:
        if reaction.id not in active_reactions:
            reactions_to_remove.append(reaction)

    print(f"Removing {len(reactions_to_remove)} reactions from model...")
    model.remove_reactions(reactions_to_remove, remove_orphans=True)

    print(
        f"Reduced model: {len(model.reactions)} reactions (from {original_reactions})"
    )
    print(f"Reduced model: {len(model.metabolites)} metabolites")
    print(f"Reduced model: {len(model.genes)} genes")

    # Save reduced model
    cobra.io.write_sbml_model(model, output_model_path)
    print(f"Reduced model saved to: {output_model_path}")

    return model, mean_presence, imat_df


def summarize_imat_analysis(csv_file_path, output_dir):
    """
    Create summary plots and statistics from iMAT solutions.
    """

    # Load iMAT data
    imat_df = pd.read_csv(csv_file_path, index_col=0, header=None)
    imat_matrix = imat_df.values

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Calculate reaction presence statistics (iMAT solutions are already binary)
    mean_presence = np.mean(imat_matrix, axis=1)

    # 1. Histogram of reaction presence
    plt.figure(figsize=(10, 6))
    plt.hist(mean_presence, bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("Fraction of samples where reaction is selected by iMAT")
    plt.ylabel("Number of reactions")
    plt.title("Distribution of Reaction Selection by iMAT Across Samples")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_dir, "imat_reaction_presence_histogram.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Summary statistics
    stats = {
        "total_reactions": len(imat_df.index),
        "total_samples": imat_df.shape[1],
        "always_selected_reactions": np.sum(mean_presence == 1.0),
        "never_selected_reactions": np.sum(mean_presence == 0.0),
        "sometimes_selected_reactions": np.sum(
            (mean_presence > 0.0) & (mean_presence < 1.0)
        ),
        "mean_selection_rate": np.mean(mean_presence),
        "median_selection_rate": np.median(mean_presence),
        "reactions_selected_in_majority": np.sum(mean_presence > 0.5),
    }

    # Save statistics
    stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
    stats_df.to_csv(os.path.join(output_dir, "imat_summary_stats.csv"), index=False)

    # 3. Reaction presence by threshold
    thresholds = np.arange(0, 1.1, 0.1)
    reactions_above_threshold = [np.sum(mean_presence >= t) for t in thresholds]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, reactions_above_threshold, "o-", linewidth=2, markersize=8)
    plt.xlabel("Selection threshold")
    plt.ylabel("Number of reactions")
    plt.title("Number of Reactions Above Different Selection Thresholds")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_dir, "imat_reactions_by_threshold.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 4. Top and bottom selected reactions
    reaction_presence_df = pd.DataFrame(
        {"reaction_id": imat_df.index, "selection_fraction": mean_presence}
    ).sort_values("selection_fraction", ascending=False)

    # Save top 50 most and least selected reactions
    reaction_presence_df.head(50).to_csv(
        os.path.join(output_dir, "top_50_selected_reactions.csv"), index=False
    )
    reaction_presence_df.tail(50).to_csv(
        os.path.join(output_dir, "bottom_50_selected_reactions.csv"), index=False
    )

    # 5. Sample-wise statistics
    sample_selection_counts = np.sum(imat_matrix, axis=0)

    plt.figure(figsize=(12, 6))
    plt.hist(sample_selection_counts, bins=30, alpha=0.7, edgecolor="black")
    plt.xlabel("Number of reactions selected per sample")
    plt.ylabel("Number of samples")
    plt.title("Distribution of Number of Reactions Selected per Sample")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_dir, "imat_reactions_per_sample.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Save sample statistics
    sample_stats = pd.DataFrame(
        {
            "sample_index": range(len(sample_selection_counts)),
            "reactions_selected": sample_selection_counts,
            "selection_percentage": (sample_selection_counts / len(imat_df.index))
            * 100,
        }
    )
    sample_stats.to_csv(
        os.path.join(output_dir, "sample_selection_stats.csv"), index=False
    )

    print(f"iMAT summary analysis saved to: {output_dir}")
    print(f"Total reactions: {stats['total_reactions']}")
    print(f"Always selected: {stats['always_selected_reactions']}")
    print(f"Never selected: {stats['never_selected_reactions']}")
    print(f"Sometimes selected: {stats['sometimes_selected_reactions']}")
    print(f"Mean selection rate: {stats['mean_selection_rate']:.3f}")
    print(f"Mean reactions per sample: {np.mean(sample_selection_counts):.1f}")

    return stats, reaction_presence_df


def compare_gimme_imat(gimme_csv_path, imat_csv_path, output_dir):
    """
    Compare GIMME and iMAT results.
    """

    if not os.path.exists(gimme_csv_path) or not os.path.exists(imat_csv_path):
        print("Warning: Both GIMME and iMAT CSV files needed for comparison")
        return

    # Load both datasets
    gimme_df = pd.read_csv(gimme_csv_path, index_col=0, header=None)
    imat_df = pd.read_csv(imat_csv_path, index_col=0, header=None)

    # Find common reactions
    common_reactions = set(gimme_df.index) & set(imat_df.index)
    print(f"Common reactions between GIMME and iMAT: {len(common_reactions)}")

    if len(common_reactions) == 0:
        print("No common reactions found between GIMME and iMAT results")
        return

    # Filter to common reactions
    gimme_common = gimme_df.loc[list(common_reactions)]
    imat_common = imat_df.loc[list(common_reactions)]

    # Convert GIMME to binary (like iMAT)
    gimme_binary = np.where(np.abs(gimme_common.values) < 1e-9, 0, 1)
    imat_binary = imat_common.values

    # Calculate presence rates
    gimme_presence = np.mean(gimme_binary, axis=1)
    imat_presence = np.mean(imat_binary, axis=1)

    # Comparison plot
    plt.figure(figsize=(10, 8))
    plt.scatter(gimme_presence, imat_presence, alpha=0.6, s=20)
    plt.xlabel("GIMME reaction presence")
    plt.ylabel("iMAT reaction presence")
    plt.title("Comparison of Reaction Presence: GIMME vs iMAT")
    plt.plot([0, 1], [0, 1], "r--", alpha=0.8, label="Perfect agreement")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_dir, "gimme_vs_imat_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Correlation
    correlation = np.corrcoef(gimme_presence, imat_presence)[0, 1]
    print(f"Correlation between GIMME and iMAT presence: {correlation:.3f}")

    # Save comparison data
    comparison_df = pd.DataFrame(
        {
            "reaction_id": list(common_reactions),
            "gimme_presence": gimme_presence,
            "imat_presence": imat_presence,
            "difference": gimme_presence - imat_presence,
        }
    ).sort_values("difference", key=abs, ascending=False)

    comparison_df.to_csv(
        os.path.join(output_dir, "gimme_imat_comparison.csv"), index=False
    )

    return comparison_df


def main():
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    csv_file_path = os.path.join(
        project_root, "files", "allsolutions_imat_tutorial.csv"
    )
    original_model_path = os.path.join(project_root, "models", "endoA_250904_clean.xml")
    output_model_path = os.path.join(
        project_root, "models", "model_reduced_imat_tutorial.xml"
    )
    summary_output_dir = os.path.join(project_root, "files", "imat_analysis_summary")

    # Check if CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        return

    # Create summary analysis
    print("=== Creating iMAT Analysis Summary ===")
    stats, reaction_df = summarize_imat_analysis(csv_file_path, summary_output_dir)

    # Create reduced model with different thresholds
    thresholds = [0.05, 0.1, 0.2, 0.5]

    for threshold in thresholds:
        print(f"\n=== Creating Reduced Model (threshold={threshold}) ===")
        threshold_model_path = output_model_path.replace(
            ".xml", f"_threshold_{threshold}.xml"
        )

        reduced_model, mean_presence, imat_df = create_reduced_model_from_imat_csv(
            csv_file_path, original_model_path, threshold_model_path, threshold
        )

        print(f"Model with {threshold*100}% threshold saved to: {threshold_model_path}")

    # Compare with GIMME if available
    gimme_csv_path = os.path.join(
        project_root, "files", "allsolutions_gimme_tutorial.csv"
    )
    if os.path.exists(gimme_csv_path):
        print("\n=== Comparing GIMME and iMAT Results ===")
        compare_gimme_imat(gimme_csv_path, csv_file_path, summary_output_dir)


if __name__ == "__main__":
    main()
