import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from typing import Dict, Any, Tuple, Optional, List
from scipy.stats import levy_stable # Import Levy Stable distribution
from decimal import Decimal, InvalidOperation # Added InvalidOperation

logger = logging.getLogger(__name__)

# --- Helper: Plot Goal Matrix (Uses MC Scores) ---
def _plot_goal_matrix_on_ax(ax, fixture_results: Dict[str, Any], max_goals: int):
    """Draws the goal matrix heatmap onto a given matplotlib Axes object based on MC results."""
    home_team = fixture_results.get("home_team", "Home")
    away_team = fixture_results.get("away_team", "Away")
    mc_score_probs = fixture_results.get("mc_score_probs") # Use mc_score_probs directly

    if not mc_score_probs:
        ax.text(0.5, 0.5, "MC Score Probs\nNot Available", ha='center', va='center', fontsize=10, color='red')
        ax.set_title("MC Score Matrix")
        ax.set_xticks([])
        ax.set_yticks([])
        return None # Return None as mesh is not created

    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            key = f"score_{hg}-{ag}"
            prob_matrix[hg, ag] = mc_score_probs.get(key, 0.0)

    mesh = ax.pcolormesh(prob_matrix.T, cmap="viridis", edgecolors='k', linewidth=0.5, vmin=0, vmax=max(0.01, prob_matrix.max())) # Ensure vmax > 0

    ax.set_xticks(np.arange(prob_matrix.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(prob_matrix.shape[1]) + 0.5, minor=False)
    ax.set_xticklabels(np.arange(prob_matrix.shape[0]))
    ax.set_yticklabels(np.arange(prob_matrix.shape[1]))

    ax.set_xlabel(f'{home_team} Goals')
    ax.set_ylabel(f'{away_team} Goals')
    ax.set_title("MC Score Probability Matrix")

    for hg in range(prob_matrix.shape[0]):
        for ag in range(prob_matrix.shape[1]):
            prob = prob_matrix[hg, ag]
            if prob > 0.001:
                v_min, v_max = mesh.norm.vmin, mesh.norm.vmax
                norm_prob = (prob - v_min) / (v_max - v_min) if (v_max - v_min) > 0 else 0
                rgba = mesh.cmap(norm_prob)
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.5 else "black"
                ax.text(hg + 0.5, ag + 0.5, f"{prob:.3f}", ha="center", va="center", color=text_color, fontsize=8)

    return mesh # Return mesh for potential colorbar


# --- Helper: Plot 1X2 Comparison ---
def _plot_1x2_comparison_on_ax(ax, fixture_results: Dict[str, Any]):
    """Draws a comparison of 1X2 probabilities from different models."""
    models = []
    probs_h, probs_d, probs_a = [], [], []
    colors = plt.cm.get_cmap('tab10', 4) # Get 4 distinct colors

    # 1. Monte Carlo
    mc_probs = fixture_results.get("mc_probs")
    if mc_probs:
        models.append("MC")
        probs_h.append(mc_probs.get('prob_H', 0))
        probs_d.append(mc_probs.get('prob_D', 0))
        probs_a.append(mc_probs.get('prob_A', 0))

    # 2. Analytical Poisson
    ap_probs = fixture_results.get("analytical_poisson_probs")
    if ap_probs:
        models.append("Analytic Poisson")
        probs_h.append(ap_probs.get('poisson_prob_H', 0))
        probs_d.append(ap_probs.get('poisson_prob_D', 0))
        probs_a.append(ap_probs.get('poisson_prob_A', 0))

    # 3. Bivariate Poisson
    bp_probs = fixture_results.get("bivariate_poisson_probs")
    if bp_probs:
        models.append("Bivar Poisson")
        probs_h.append(bp_probs.get('biv_poisson_prob_H', 0))
        probs_d.append(bp_probs.get('biv_poisson_prob_D', 0))
        probs_a.append(bp_probs.get('biv_poisson_prob_A', 0))

    # 4. Elo
    elo_probs = fixture_results.get("elo_probs")
    if elo_probs and elo_probs.get('elo_prob_H') is not None: # Check for valid Elo results
        models.append("Elo")
        probs_h.append(elo_probs.get('elo_prob_H', 0))
        probs_d.append(elo_probs.get('elo_prob_D', 0))
        probs_a.append(elo_probs.get('elo_prob_A', 0))

    if not models:
        ax.text(0.5, 0.5, "No 1X2 Probs Available", ha='center', va='center', fontsize=10, color='red')
        ax.set_title("Model Comparison: 1X2")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    outcomes = ['Home Win', 'Draw', 'Away Win']
    n_models = len(models)
    n_outcomes = len(outcomes)
    x = np.arange(n_outcomes) # Positions for outcomes
    width = 0.8 / n_models # Adjust bar width based on number of models

    all_rects = []
    for i, model in enumerate(models):
        model_probs = [probs_h[i], probs_d[i], probs_a[i]]
        offset = (i - (n_models - 1) / 2) * width # Center the group of bars
        rects = ax.bar(x + offset, model_probs, width, label=model, color=colors(i))
        all_rects.append(rects)

    ax.set_ylabel('Probability')
    ax.set_title('Model Comparison: 1X2 Probabilities')
    ax.set_xticks(x)
    ax.set_xticklabels(outcomes)
    ax.legend(fontsize=8, loc='upper right', ncol=min(n_models, 2)) # Adjust legend position and columns
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Add labels to bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=6) # Smaller font size for labels

    for rects in all_rects:
        add_labels(rects)


# --- Helper: Plot Market Comparison (O/U 2.5, BTTS) ---
def _plot_market_comparison_on_ax(ax, fixture_results: Dict[str, Any]):
    """Draws a comparison of O/U 2.5 and BTTS probabilities from different models."""
    models = []
    probs_o25, probs_btts_yes = [], []
    colors = plt.cm.get_cmap('tab10', 3) # Get 3 distinct colors

    # 1. Monte Carlo
    mc_probs = fixture_results.get("mc_probs")
    if mc_probs:
        models.append("MC")
        probs_o25.append(mc_probs.get('prob_O2.5', 0))
        probs_btts_yes.append(mc_probs.get('prob_BTTS Yes', 0))

    # 2. Analytical Poisson
    ap_probs = fixture_results.get("analytical_poisson_probs")
    if ap_probs:
        models.append("Analytic Poisson")
        probs_o25.append(ap_probs.get('poisson_prob_O2.5', 0))
        probs_btts_yes.append(ap_probs.get('poisson_prob_BTTS_Yes', 0))

    # 3. Bivariate Poisson
    bp_probs = fixture_results.get("bivariate_poisson_probs")
    if bp_probs:
        models.append("Bivar Poisson")
        probs_o25.append(bp_probs.get('biv_poisson_prob_O2.5', 0))
        probs_btts_yes.append(bp_probs.get('biv_poisson_prob_BTTS_Yes', 0))

    if not models:
        ax.text(0.5, 0.5, "No Market Probs Available", ha='center', va='center', fontsize=10, color='red')
        ax.set_title("Model Comparison: O/U 2.5 & BTTS")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    markets = ['Over 2.5 Goals', 'BTTS Yes']
    n_models = len(models)
    n_markets = len(markets)
    x = np.arange(n_markets) # Positions for markets
    width = 0.8 / n_models # Adjust bar width

    all_rects = []
    for i, model in enumerate(models):
        model_probs = [probs_o25[i], probs_btts_yes[i]]
        offset = (i - (n_models - 1) / 2) * width # Center the group
        rects = ax.bar(x + offset, model_probs, width, label=model, color=colors(i))
        all_rects.append(rects)

    ax.set_ylabel('Probability')
    ax.set_title('Model Comparison: O/U 2.5 & BTTS Probabilities')
    ax.set_xticks(x)
    ax.set_xticklabels(markets)
    ax.legend(fontsize=8, loc='upper right', ncol=min(n_models, 2))
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Add labels
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7) # Slightly smaller font size

    for rects in all_rects:
        add_labels(rects)


# --- Helper: Plot Goal Distribution (Levy PDF using Original Lambdas) ---
def _plot_goal_pdfs_on_ax(ax, fixture_results: Dict[str, Any], max_goals_axis: int = 8):
    """Draws the Levy Stable goal distribution PDFs using the original lambdas."""
    # IMPORTANT: Use 'lambdas_original' which are used for MC/Poisson/Bivar calculations
    lambdas_orig: Optional[Tuple[Optional[float], Optional[float]]] = fixture_results.get("lambdas_original")
    home_team = fixture_results.get("home_team", "Home")
    away_team = fixture_results.get("away_team", "Away")

    if lambdas_orig is None or lambdas_orig[0] is None or lambdas_orig[1] is None:
        ax.text(0.5, 0.5, "Original Lambdas (xG)\nNot Available", ha='center', va='center', fontsize=10, color='red')
        ax.set_title("Goal Distribution PDF (Levy)")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    lambda_home, lambda_away = lambdas_orig
    lambda_home = max(0.01, lambda_home) # Ensure lambdas > 0 for sqrt/log etc.
    lambda_away = max(0.01, lambda_away)

    # Define Levy Stable parameters
    alpha = 1.6  # Stability parameter (tail heaviness)
    beta = 0.0   # Skewness parameter (0 = symmetric)
    scale_home = 0.5 + 0.25 * np.sqrt(lambda_home) # Heuristic scale
    scale_away = 0.5 + 0.25 * np.sqrt(lambda_away)

    x_goals = np.linspace(-0.5, max_goals_axis + 0.5, 200)

    try:
        pdf_home = levy_stable.pdf(x_goals, alpha, beta, loc=lambda_home, scale=scale_home)
        pdf_away = levy_stable.pdf(x_goals, alpha, beta, loc=lambda_away, scale=scale_away)
    except Exception as e:
         logger.error(f"Error calculating Levy PDF: {e}", exc_info=True)
         ax.text(0.5, 0.5, "Levy PDF Calculation\nError", ha='center', va='center', fontsize=10, color='red')
         ax.set_title("Goal Distribution PDF (Levy)")
         return

    ax.plot(x_goals, pdf_home, label=f'{home_team} (λ={lambda_home:.2f})', color='#1f77b4', linewidth=2)
    ax.plot(x_goals, pdf_away, label=f'{away_team} (λ={lambda_away:.2f})', color='#d62728', linewidth=2)

    ax.set_xlabel('Number of Goals')
    ax.set_ylabel('Probability Density')
    ax.set_title('Goal Distribution PDF (Levy Stable)')
    ax.legend(fontsize=8)
    ax.set_xlim(left=-0.5, right=max_goals_axis + 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.6)

    param_text = (f"Levy (α={alpha:.1f}, β={beta:.1f})\n"
                  f"H: loc={lambda_home:.2f}, sc={scale_home:.2f}\n"
                  f"A: loc={lambda_away:.2f}, sc={scale_away:.2f}")
    ax.annotate(param_text, xy=(0.97, 0.97), xycoords='axes fraction',
                verticalalignment='top', horizontalalignment='right', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))


# --- Main Plotting Function ---
def create_combined_fixture_plot(
    fixture_results: Dict[str, Any],
    output_dir: str,
    max_goals_matrix: int = 5,
    max_goals_pdf_axis: int = 8 # Max goals for PDF plot axis
):
    """
    Generates and saves a single 2x2 plot combining key fixture insights and model comparisons.
    """
    if not fixture_results:
        logger.warning("create_combined_fixture_plot received empty fixture_results.")
        return

    fixture_id = fixture_results.get("fixture_id", "unknown")
    home_team = fixture_results.get("home_team", "Home")
    away_team = fixture_results.get("away_team", "Away")

    # Sanitize filename
    plot_filename_base = f"fixture_{fixture_id}_{home_team}_vs_{away_team}_Analysis"
    plot_filename_base = "".join(c if c.isalnum() else "_" for c in plot_filename_base).replace("__", "_")
    plot_path = os.path.join(output_dir, f"{plot_filename_base}.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # Increased figsize for better spacing with more complex plots

    # --- Draw each subplot ---
    # Top-Left: MC Score Matrix
    mesh = _plot_goal_matrix_on_ax(axes[0, 0], fixture_results, max_goals_matrix)
    if mesh: # Add colorbar if matrix was plotted
        fig.colorbar(mesh, ax=axes[0, 0], label='MC Score Probability', fraction=0.046, pad=0.04)

    # Top-Right: 1X2 Comparison
    _plot_1x2_comparison_on_ax(axes[0, 1], fixture_results)

    # Bottom-Left: Market Comparison (O/U 2.5, BTTS)
    _plot_market_comparison_on_ax(axes[1, 0], fixture_results)

    # Bottom-Right: Goal Distribution PDF (using original lambdas)
    _plot_goal_pdfs_on_ax(axes[1, 1], fixture_results, max_goals_axis=max_goals_pdf_axis)

    # --- Overall Figure Title ---
    # Include both original and weighted lambdas in the title for comparison
    lambdas_orig: Optional[Tuple[Optional[float], Optional[float]]] = fixture_results.get("lambdas_original")
    lambdas_w: Optional[Tuple[Optional[float], Optional[float]]] = fixture_results.get("lambdas_weighted")

    lambda_h_orig_str = f"{lambdas_orig[0]:.3f}" if lambdas_orig and lambdas_orig[0] is not None else "N/A"
    lambda_a_orig_str = f"{lambdas_orig[1]:.3f}" if lambdas_orig and lambdas_orig[1] is not None else "N/A"
    lambda_h_w_str = f"{lambdas_w[0]:.3f}" if lambdas_w and lambdas_w[0] is not None else "N/A"
    lambda_a_w_str = f"{lambdas_w[1]:.3f}" if lambdas_w and lambdas_w[1] is not None else "N/A"

    title = (f'Fixture Analysis: {home_team} vs {away_team} (ID: {fixture_id})\n'
             f'Strength Lambdas (H/A): {lambda_h_orig_str} / {lambda_a_orig_str} | '
             f'Weighted Lambdas (H/A): {lambda_h_w_str} / {lambda_a_w_str}')
    fig.suptitle(title, fontsize=14)

    # --- Adjust Layout & Save ---
    try:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent title overlap
        plt.savefig(plot_path)
        logger.info(f"Saved Combined Analysis plot to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to create or save combined plot {plot_path}: {e}", exc_info=True)
    finally:
        # Ensure figure is closed regardless of saving success
        if plt.fignum_exists(fig.number):
            plt.close(fig) 


def get_metric_value(paper: Dict[str, Any], metric_key: str) -> Optional[float]:
    """Helper function to get metric values from paper data."""
    try:
        # First check paper_metrics
        if metric_key in paper.get('paper_metrics', {}):
            return float(paper['paper_metrics'][metric_key])
        # Then check top level
        elif metric_key in paper:
            return float(paper[metric_key])
        return None
    except (ValueError, TypeError):
        return None


def plot_paper_scatter_mpl(papers_data: list, output_file: str,
                           x_axis_metric: str, y_axis_metric: str,
                           color_metric: str = None, size_metric: str = None,
                           title: str = "Betting Paper Analysis",
                           default_size: int = 30, size_multiplier: float = 500.0,
                           x_log: bool = False, y_log: bool = False):
    """
    Generates a scatter plot of betting papers using Matplotlib.

    Args:
        papers_data: List of paper dictionaries (output from generate_papers).
        output_file: Path to save the generated PNG plot.
        x_axis_metric: Key for the metric on the x-axis (from paper_metrics or top-level).
        y_axis_metric: Key for the metric on the y-axis.
        color_metric: Key for the metric determining marker color (optional).
        size_metric: Key for the metric determining marker size (optional).
        title: Title for the plot.
        default_size: Default marker size if size_metric is not used or invalid.
        size_multiplier: Factor to scale the size_metric values for visibility.
        x_log: Whether to use a logarithmic scale for the x-axis.
        y_log: Whether to use a logarithmic scale for the y-axis.
    """
    if not papers_data:
        logger.warning("No paper data provided for plotting. Skipping plot generation.")
        return

    x_values, y_values, colors, sizes = [], [], [], []
    valid_points = 0

    for paper in papers_data:
        x = get_metric_value(paper, x_axis_metric)
        y = get_metric_value(paper, y_axis_metric)

        # Skip points with invalid essential coordinates
        if x is None or y is None:
            logger.debug(f"Skipping point for paper {paper.get('paper_id')}: Missing X ({x_axis_metric}={x}) or Y ({y_axis_metric}={y}).")
            continue
        # Skip points if log scale is requested and value is non-positive
        if (x_log and x <= 0) or (y_log and y <= 0):
            logger.debug(f"Skipping point for paper {paper.get('paper_id')} due to non-positive value for log scale axis (X:{x}, Y:{y}).")
            continue

        x_values.append(x)
        y_values.append(y)

        # Determine color
        c_val = None
        if color_metric:
            c_val = get_metric_value(paper, color_metric)
            # Handle cases like 'num_legs' which might be discrete
            if c_val is None and color_metric == 'num_legs':
                c_val = paper.get('paper_metrics', {}).get('num_legs') # Get directly if metric helper fails

        colors.append(c_val if c_val is not None else np.nan) # Use NaN for missing color values

        # Determine size
        s_val = None
        if size_metric:
            s_val = get_metric_value(paper, size_metric)
        # Scale size, use default if invalid/missing, ensure non-negative
        size = default_size
        if s_val is not None and s_val > 0:
            size = max(5, s_val * size_multiplier) # Ensure a minimum size, scale positive values
        elif s_val is not None and s_val <= 0 and size_metric == 'average_edge':
            # Special handling: small size for zero/negative edge
            size = 5
        sizes.append(size)
        valid_points += 1

    if not valid_points:
        logger.warning(f"No valid data points found for plotting '{title}'. Skipping plot generation.")
        return

    logger.info(f"Plotting {valid_points} valid data points for '{title}'.")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Choose colormap - Viridis is good for sequential data, coolwarm for diverging (like edge)
    cmap = 'viridis'
    norm = None
    scatter_args = {'alpha': 0.7}

    # Handle color mapping
    valid_colors = [c for c in colors if c is not None and not np.isnan(c)]
    if color_metric and valid_colors:
        min_c, max_c = min(valid_colors), max(valid_colors)
        # Discrete colors for num_legs (or similar low-cardinality metrics)
        if color_metric == 'num_legs' and len(set(valid_colors)) < 10:
            unique_legs = sorted(list(set(int(l) for l in valid_colors)))
            cmap = plt.get_cmap('viridis', len(unique_legs)) # Discrete colormap
            norm = plt.colors.BoundaryNorm(np.arange(min(unique_legs)-0.5, max(unique_legs)+1.5, 1), cmap.N)
            scatter_args['cmap'] = cmap
            scatter_args['norm'] = norm
        else: # Continuous color mapping
            norm = plt.colors.Normalize(vmin=min_c, vmax=max_c)
            scatter_args['cmap'] = cmap
            scatter_args['norm'] = norm
        scatter_args['c'] = colors # Pass the list of color values
    else:
        scatter_args['c'] = 'blue' # Default color if no metric or no valid data

    # Handle size mapping
    scatter_args['s'] = sizes # Pass the list of calculated sizes

    # Create the scatter plot
    sc = ax.scatter(x_values, y_values, **scatter_args)

    # Add colorbar if color metric was used and valid
    if color_metric and valid_colors:
        cbar = fig.colorbar(sc, label=color_metric.replace('_', ' ').title())
        if color_metric == 'num_legs' and isinstance(norm, plt.colors.BoundaryNorm):
            # Set ticks for discrete colorbar
            tick_locs = np.array(unique_legs)
            cbar.set_ticks(tick_locs)
            cbar.set_ticklabels(unique_legs)

    ax.set_xlabel(x_axis_metric.replace('_', ' ').title())
    ax.set_ylabel(y_axis_metric.replace('_', ' ').title())
    ax.set_title(title)

    # Apply log scale if requested
    if x_log: ax.set_xscale('log')
    if y_log: ax.set_yscale('log')

    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    try:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved successfully to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save plot {output_file}: {e}")
    finally:
        plt.close(fig) # Close the figure to free memory