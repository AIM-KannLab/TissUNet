import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_results(
    mask_slice,
    checkpoints,
    thicknesses,
    quadrant_stats,
    center_x,
    center_y,
    contour,
    z,
    filename,
    text_offset=-7,
    save_path=None,
):
    """
    Plot the binary mask, checkpoints, thickness values, and quadrant statistics.
    """
    plt.rcParams['font.family'] = 'serif'
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_slice, cmap="hot", interpolation="nearest")
    plt.title(
        f"Thickness at Checkpoints and Quadrant Aggregates (Slice {z}, File: {filename})"
    )

    # Plot the checkpoints with their thickness values, tangents, and perpendiculars
    for i, point in enumerate(checkpoints):
        x, y = point[0], point[1]
        # Plot the checkpoint
        plt.plot(x, y, "go")  # 'go' for green circles

        # Calculate and plot the tangent vector
        idx = np.argmin(
            np.sqrt((contour[:, 1] - x) ** 2 + (contour[:, 0] - y) ** 2)
        )  # Find closest contour point to checkpoint
        neighborhood_indices = np.arange(idx - 5, idx + 6) % len(contour)
        neighborhood_points = contour[neighborhood_indices]
        dx = np.mean(np.diff(neighborhood_points[:, 1]))
        dy = np.mean(np.diff(neighborhood_points[:, 0]))
        tangent = np.array([dx, dy])
        tangent /= np.linalg.norm(tangent)  # Normalize the tangent
        tangent_length = 3
        plt.arrow(
            x,
            y,
            tangent[0] * tangent_length,
            tangent[1] * tangent_length,
            color="red",
            head_width=0.01,
            head_length=0.01,
            width=0.05,
        )
        plt.arrow(
            x,
            y,
            -tangent[0] * tangent_length,
            -tangent[1] * tangent_length,
            color="red",
            head_width=0.01,
            head_length=0.01,
            width=0.05,
        )

        # Calculate and plot the perpendicular vector (orthogonal to tangent)
        perp_vector = np.array(
            [-tangent[1], tangent[0]]
        )  # Rotate tangent by 90 degrees
        plt.arrow(
            x,
            y,
            perp_vector[0] * thicknesses[i],
            perp_vector[1] * thicknesses[i],
            color="magenta",
            head_width=0.01,
            head_length=0.01,
            width=0.05,
        )

        # Position the text with an offset along the perpendicular vector
        text_x = x + perp_vector[0] * text_offset
        text_y = y + perp_vector[1] * text_offset
        plt.text(
            text_x,
            text_y,
            f"{thicknesses[i]:.2f}",
            color="white",
            fontsize=8,
            ha="center",
        )

    positions = {
        "top_left": (5, 5, "left", "top"),
        "top_right": (mask_slice.shape[1] - 5, 5, "right", "top"),
        "bottom_left": (5, mask_slice.shape[0] - 5, "left", "bottom"),
        "bottom_right": (
            mask_slice.shape[1] - 5,
            mask_slice.shape[0] - 5,
            "right",
            "bottom",
        ),
    }
    for quadrant, stats in quadrant_stats.items():
        x, y, ha, va = positions[quadrant]
        text = (
            f"Mean: {stats['mean']:.2f}\n"
            f"Median: {stats['median']:.2f}\n"
            f"2.5th Percentile: {stats['0.025 percentile']:.2f}\n"
            f"97.5th Percentile: {stats['0.975 percentile']:.2f}\n"
            f"Count: {stats['count']}"
        )
        plt.text(
            x,
            y,
            text,
            ha=ha,
            va=va,
            color="white",
            fontsize=12,
        )

    plt.axvline(x=center_x, color="white", linestyle="--", linewidth=0.5)
    plt.axhline(y=center_y, color="white", linestyle="--", linewidth=0.5)

    if save_path:  # Save the plot if a path is provided
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

