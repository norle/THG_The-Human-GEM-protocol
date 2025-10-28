import pandas as pd
import numpy as np
import cobra
import os
import matplotlib.pyplot as plt


def create_reduced_model_from_fastcore_csv(
    csv_file_path, original_model_path, output_model_path, presence_threshold=0.1
):
    """
    Create a reduced model based on reaction presence in FastCORE solutions.

    Parameters:
    - csv_file_path: Path to the CSV file with FastCORE solutions (rows=reactions, cols=samples)
    - original_model_path: Path to the original SBML model
    - output_model_path: Path to save the reduced model
    - presence_threshold: Minimum fraction of samples where reaction must be active (default 0.1 = 10%)
    """

    # Load the FastCORE solutions CSV (with reaction IDs as index)
    print("Loading FastCORE solutions...")
    fastcore_df = pd.read_csv(csv_file_path, index_col=0, header=None)
    print(
        f"Loaded FastCORE data: {fastcore_df.shape[0]} reactions × {fastcore_df.shape[1]} samples"
    )

    # Convert to numpy array for calculations
    fastcore_matrix = fastcore_df.values

    # FastCORE solutions are already binary (0 or 1), so no need to discretize
    # Calculate mean presence of each reaction across samples
    mean_presence = np.mean(fastcore_matrix, axis=1)

    # Get reaction IDs that meet the presence threshold
    active_reactions = fastcore_df.index[mean_presence >= presence_threshold].tolist()

    print(
        f"Reactions meeting {presence_threshold*100}% presence threshold: {len(active_reactions)}/{len(fastcore_df.index)}"
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

    return model, mean_presence, fastcore_df


def summarize_fastcore_analysis(csv_file_path, output_dir):
    """
    Create summary plots and statistics from FastCORE solutions.
    """

    # Load FastCORE data
    fastcore_df = pd.read_csv(csv_file_path, index_col=0, header=None)
    fastcore_matrix = fastcore_df.values

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Calculate reaction presence statistics (FastCORE solutions are already binary)
    mean_presence = np.mean(fastcore_matrix, axis=1)

    # 1. Histogram of reaction presence
    plt.figure(figsize=(10, 6))
    plt.hist(mean_presence, bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("Fraction of samples where reaction is included by FastCORE")
    plt.ylabel("Number of reactions")
    plt.title("Distribution of Reaction Inclusion by FastCORE Across Samples")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_dir, "fastcore_reaction_presence_histogram.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Summary statistics
    stats = {
        "total_reactions": len(fastcore_df.index),
        "total_samples": fastcore_df.shape[1],
        "always_included_reactions": np.sum(mean_presence == 1.0),
        "never_included_reactions": np.sum(mean_presence == 0.0),
        "sometimes_included_reactions": np.sum(
            (mean_presence > 0.0) & (mean_presence < 1.0)
        ),
        "mean_inclusion_rate": np.mean(mean_presence),
        "median_inclusion_rate": np.median(mean_presence),
        "reactions_included_in_majority": np.sum(mean_presence > 0.5),
    }

    # Save statistics
    stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
    stats_df.to_csv(os.path.join(output_dir, "fastcore_summary_stats.csv"), index=False)

    # 3. Reaction presence by threshold
    thresholds = np.arange(0, 1.1, 0.1)
    reactions_above_threshold = [np.sum(mean_presence >= t) for t in thresholds]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, reactions_above_threshold, "o-", linewidth=2, markersize=8)
    plt.xlabel("Inclusion threshold")
    plt.ylabel("Number of reactions")
    plt.title("Number of Reactions Above Different Inclusion Thresholds")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_dir, "fastcore_reactions_by_threshold.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 4. Top and bottom included reactions
    reaction_presence_df = pd.DataFrame(
        {"reaction_id": fastcore_df.index, "inclusion_fraction": mean_presence}
    ).sort_values("inclusion_fraction", ascending=False)

    # Save top 50 most and least included reactions
    reaction_presence_df.head(50).to_csv(
        os.path.join(output_dir, "top_50_included_reactions.csv"), index=False
    )
    reaction_presence_df.tail(50).to_csv(
        os.path.join(output_dir, "bottom_50_included_reactions.csv"), index=False
    )

    # 5. Sample-wise statistics
    sample_inclusion_counts = np.sum(fastcore_matrix, axis=0)

    plt.figure(figsize=(12, 6))
    plt.hist(sample_inclusion_counts, bins=30, alpha=0.7, edgecolor="black")
    plt.xlabel("Number of reactions included per sample")
    plt.ylabel("Number of samples")
    plt.title("Distribution of Number of Reactions Included per Sample")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_dir, "fastcore_reactions_per_sample.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Save sample statistics
    sample_stats = pd.DataFrame(
        {
            "sample_index": range(len(sample_inclusion_counts)),
            "reactions_included": sample_inclusion_counts,
            "inclusion_percentage": (sample_inclusion_counts / len(fastcore_df.index))
            * 100,
        }
    )
    sample_stats.to_csv(
        os.path.join(output_dir, "sample_inclusion_stats.csv"), index=False
    )

    print(f"FastCORE summary analysis saved to: {output_dir}")
    print(f"Total reactions: {stats['total_reactions']}")
    print(f"Always included: {stats['always_included_reactions']}")
    print(f"Never included: {stats['never_included_reactions']}")
    print(f"Sometimes included: {stats['sometimes_included_reactions']}")
    print(f"Mean inclusion rate: {stats['mean_inclusion_rate']:.3f}")
    print(f"Mean reactions per sample: {np.mean(sample_inclusion_counts):.1f}")

    return stats, reaction_presence_df


def compare_methods(gimme_csv_path, imat_csv_path, fastcore_csv_path, output_dir):
    """
    Compare GIMME, iMAT, and FastCORE results.
    """

    methods = {}
    if os.path.exists(gimme_csv_path):
        methods["GIMME"] = pd.read_csv(gimme_csv_path, index_col=0, header=None)
    if os.path.exists(imat_csv_path):
        methods["iMAT"] = pd.read_csv(imat_csv_path, index_col=0, header=None)
    if os.path.exists(fastcore_csv_path):
        methods["FastCORE"] = pd.read_csv(fastcore_csv_path, index_col=0, header=None)

    if len(methods) < 2:
        print("Need at least 2 method results for comparison")
        return

    # Find common reactions
    common_reactions = set(list(methods.values())[0].index)
    for df in methods.values():
        common_reactions &= set(df.index)

    print(f"Common reactions across all methods: {len(common_reactions)}")

    if len(common_reactions) == 0:
        print("No common reactions found across methods")
        return

    # Calculate presence rates for each method
    presence_data = {}
    for method_name, df in methods.items():
        df_common = df.loc[list(common_reactions)]
        if method_name == "GIMME":
            # Convert GIMME flux to binary
            binary_matrix = np.where(np.abs(df_common.values) < 1e-9, 0, 1)
        else:
            # iMAT and FastCORE are already binary
            binary_matrix = df_common.values
        presence_data[method_name] = np.mean(binary_matrix, axis=1)

    # Create comparison plots
    method_names = list(presence_data.keys())
    if len(method_names) == 2:
        # Scatter plot for 2 methods
        plt.figure(figsize=(10, 8))
        plt.scatter(
            presence_data[method_names[0]],
            presence_data[method_names[1]],
            alpha=0.6,
            s=20,
        )
        plt.xlabel(f"{method_names[0]} reaction presence")
        plt.ylabel(f"{method_names[1]} reaction presence")
        plt.title(f"Comparison: {method_names[0]} vs {method_names[1]}")
        plt.plot([0, 1], [0, 1], "r--", alpha=0.8, label="Perfect agreement")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(
                output_dir,
                f"{method_names[0].lower()}_vs_{method_names[1].lower()}_comparison.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Calculate correlation
        correlation = np.corrcoef(
            presence_data[method_names[0]], presence_data[method_names[1]]
        )[0, 1]
        print(
            f"Correlation between {method_names[0]} and {method_names[1]}: {correlation:.3f}"
        )

    elif len(method_names) == 3:
        # Create 3-way comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        pairs = [(0, 1), (0, 2), (1, 2)]
        for i, (idx1, idx2) in enumerate(pairs):
            name1, name2 = method_names[idx1], method_names[idx2]
            axes[i].scatter(presence_data[name1], presence_data[name2], alpha=0.6, s=20)
            axes[i].set_xlabel(f"{name1} presence")
            axes[i].set_ylabel(f"{name2} presence")
            axes[i].set_title(f"{name1} vs {name2}")
            axes[i].plot([0, 1], [0, 1], "r--", alpha=0.8)
            axes[i].grid(True, alpha=0.3)

            # Calculate correlation
            corr = np.corrcoef(presence_data[name1], presence_data[name2])[0, 1]
            axes[i].text(
                0.05,
                0.95,
                f"r = {corr:.3f}",
                transform=axes[i].transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "three_method_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Save comparison data
    comparison_df = pd.DataFrame(
        {
            "reaction_id": list(common_reactions),
            **{method: presence_data[method] for method in method_names},
        }
    )
    comparison_df.to_csv(os.path.join(output_dir, "method_comparison.csv"), index=False)

    return comparison_df


def main():
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    csv_file_path = os.path.join(
        project_root, "files", "allsolutions_fastcore_tutorial.csv"
    )
    original_model_path = os.path.join(project_root, "models", "endoA_250904_clean.xml")
    output_model_path = os.path.join(
        project_root, "models", "model_reduced_fastcore_tutorial.xml"
    )
    summary_output_dir = os.path.join(
        project_root, "files", "fastcore_analysis_summary"
    )

    # Check if CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        return

    # Create summary analysis
    print("=== Creating FastCORE Analysis Summary ===")
    stats, reaction_df = summarize_fastcore_analysis(csv_file_path, summary_output_dir)

    # Create reduced model with different thresholds
    thresholds = [0.05, 0.1, 0.2, 0.5]

    for threshold in thresholds:
        print(f"\n=== Creating Reduced Model (threshold={threshold}) ===")
        threshold_model_path = output_model_path.replace(
            ".xml", f"_threshold_{threshold}.xml"
        )

        reduced_model, mean_presence, fastcore_df = (
            create_reduced_model_from_fastcore_csv(
                csv_file_path, original_model_path, threshold_model_path, threshold
            )
        )

        print(f"Model with {threshold*100}% threshold saved to: {threshold_model_path}")

    # Compare with other methods if available
    gimme_csv_path = os.path.join(
        project_root, "files", "allsolutions_gimme_tutorial.csv"
    )
    imat_csv_path = os.path.join(
        project_root, "files", "allsolutions_imat_tutorial.csv"
    )

    available_methods = []
    if os.path.exists(gimme_csv_path):
        available_methods.append("GIMME")
    if os.path.exists(imat_csv_path):
        available_methods.append("iMAT")
    if os.path.exists(csv_file_path):
        available_methods.append("FastCORE")

    if len(available_methods) > 1:
        print(f"\n=== Comparing Methods: {', '.join(available_methods)} ===")
        compare_methods(
            gimme_csv_path, imat_csv_path, csv_file_path, summary_output_dir
        )


if __name__ == "__main__":
    main()
