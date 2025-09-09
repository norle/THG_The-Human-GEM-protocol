import pandas as pd
import numpy as np
import cobra
import os
import matplotlib.pyplot as plt


def create_reduced_model_from_csv(
    csv_file_path, original_model_path, output_model_path, presence_threshold=0.1
):
    """
    Create a reduced model based on reaction presence in flux solutions.

    Parameters:
    - csv_file_path: Path to the CSV file with flux solutions (rows=reactions, cols=samples)
    - original_model_path: Path to the original SBML model
    - output_model_path: Path to save the reduced model
    - presence_threshold: Minimum fraction of samples where reaction must be active (default 0.1 = 10%)
    """

    # Load the flux solutions CSV (with reaction IDs as index)
    print("Loading flux solutions...")
    flux_df = pd.read_csv(csv_file_path, index_col=0, header=None)
    print(
        f"Loaded flux data: {flux_df.shape[0]} reactions × {flux_df.shape[1]} samples"
    )

    # Convert to numpy array for calculations
    flux_matrix = flux_df.values

    # Discretize the matrix: 0 if flux is 0, 1 if flux is not 0
    bool_matrix = np.where(np.abs(flux_matrix) < 1e-9, 0, 1)

    # Calculate mean presence of each reaction across samples
    mean_presence = np.mean(bool_matrix, axis=1)

    # Get reaction IDs that meet the presence threshold
    active_reactions = flux_df.index[mean_presence >= presence_threshold].tolist()

    print(
        f"Reactions meeting {presence_threshold*100}% presence threshold: {len(active_reactions)}/{len(flux_df.index)}"
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

    return model, mean_presence, flux_df


def summarize_flux_analysis(csv_file_path, output_dir):
    """
    Create summary plots and statistics from flux solutions.
    """

    # Load flux data
    flux_df = pd.read_csv(csv_file_path, index_col=0, header=None)
    flux_matrix = flux_df.values

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Calculate reaction presence statistics
    bool_matrix = np.where(np.abs(flux_matrix) < 1e-9, 0, 1)
    mean_presence = np.mean(bool_matrix, axis=1)

    # 1. Histogram of reaction presence
    plt.figure(figsize=(10, 6))
    plt.hist(mean_presence, bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("Fraction of samples where reaction is active")
    plt.ylabel("Number of reactions")
    plt.title("Distribution of Reaction Activity Across Samples")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_dir, "reaction_presence_histogram.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Summary statistics
    stats = {
        "total_reactions": len(flux_df.index),
        "total_samples": flux_df.shape[1],
        "always_active_reactions": np.sum(mean_presence == 1.0),
        "never_active_reactions": np.sum(mean_presence == 0.0),
        "sometimes_active_reactions": np.sum(
            (mean_presence > 0.0) & (mean_presence < 1.0)
        ),
        "mean_activity_rate": np.mean(mean_presence),
        "median_activity_rate": np.median(mean_presence),
    }

    # Save statistics
    stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
    stats_df.to_csv(os.path.join(output_dir, "flux_summary_stats.csv"), index=False)

    # 3. Reaction presence by threshold
    thresholds = np.arange(0, 1.1, 0.1)
    reactions_above_threshold = [np.sum(mean_presence >= t) for t in thresholds]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, reactions_above_threshold, "o-", linewidth=2, markersize=8)
    plt.xlabel("Presence threshold")
    plt.ylabel("Number of reactions")
    plt.title("Number of Reactions Above Different Presence Thresholds")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_dir, "reactions_by_threshold.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 4. Top and bottom active reactions
    reaction_presence_df = pd.DataFrame(
        {"reaction_id": flux_df.index, "presence_fraction": mean_presence}
    ).sort_values("presence_fraction", ascending=False)

    # Save top 50 most and least active reactions
    reaction_presence_df.head(50).to_csv(
        os.path.join(output_dir, "top_50_active_reactions.csv"), index=False
    )
    reaction_presence_df.tail(50).to_csv(
        os.path.join(output_dir, "bottom_50_active_reactions.csv"), index=False
    )

    print(f"Summary analysis saved to: {output_dir}")
    print(f"Total reactions: {stats['total_reactions']}")
    print(f"Always active: {stats['always_active_reactions']}")
    print(f"Never active: {stats['never_active_reactions']}")
    print(f"Sometimes active: {stats['sometimes_active_reactions']}")
    print(f"Mean activity rate: {stats['mean_activity_rate']:.3f}")

    return stats, reaction_presence_df


def main():
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    csv_file_path = os.path.join(
        project_root, "files", "allsolutions_gimme_tutorial.csv"
    )
    original_model_path = os.path.join(project_root, "models", "endoA_250904_clean.xml")
    output_model_path = os.path.join(
        project_root, "models", "model_reduced_gimme_tutorial.xml"
    )
    summary_output_dir = os.path.join(project_root, "files", "gimme_analysis_summary")

    # Check if CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        return

    # Create summary analysis
    print("=== Creating Flux Analysis Summary ===")
    stats, reaction_df = summarize_flux_analysis(csv_file_path, summary_output_dir)

    # Create reduced model with different thresholds
    thresholds = [0.05, 0.1, 0.2, 0.5]

    for threshold in thresholds:
        print(f"\n=== Creating Reduced Model (threshold={threshold}) ===")
        threshold_model_path = output_model_path.replace(
            ".xml", f"_threshold_{threshold}.xml"
        )

        reduced_model, mean_presence, flux_df = create_reduced_model_from_csv(
            csv_file_path, original_model_path, threshold_model_path, threshold
        )

        print(f"Model with {threshold*100}% threshold saved to: {threshold_model_path}")


if __name__ == "__main__":
    main()
