import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from typing import Dict, Any
from scipy.stats import levy_stable # Import Levy Stable distribution

logger = logging.getLogger(__name__)

# --- Helper: Plot Goal Matrix ---
def _plot_goal_matrix_on_ax(ax, fixture_results: Dict[str, Any], max_goals: int):
    """Draws the goal matrix heatmap onto a given matplotlib Axes object."""
    home_team = fixture_results.get("home_team", "Home")
    away_team = fixture_results.get("away_team", "Away")
    mc_score_probs = fixture_results.get("mc_score_probs") # Use mc_score_probs directly

    if not mc_score_probs:
        ax.text(0.5, 0.5, "MC Score Probs\nNot Available", ha='center', va='center', fontsize=10, color='red')
        ax.set_title("MC Score Matrix")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            key = f"score_{hg}-{ag}"
            # Use mc_score_probs dict structure directly
            prob_matrix[hg, ag] = mc_score_probs.get(key, 0.0)

    # Use a perceptually uniform colormap like 'viridis' or 'plasma'
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
                # Normalize prob for color check - handle zero range
                v_min, v_max = mesh.norm.vmin, mesh.norm.vmax
                norm_prob = (prob - v_min) / (v_max - v_min) if (v_max - v_min) > 0 else 0
                # Determine text color based on background luminance
                rgba = mesh.cmap(norm_prob)
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.5 else "black"
                ax.text(hg + 0.5, ag + 0.5, f"{prob:.3f}", ha="center", va="center", color=text_color, fontsize=8) # Reduced font size

    # Return mesh for potential colorbar
    return mesh


# --- Helper: Plot MC vs Podos Comparison ---
def _plot_mc_vs_podos_on_ax(ax, fixture_results: Dict[str, Any]):
    """Draws the MC vs Podos 1X2 comparison chart onto a given Axes object."""
    mc_probs = fixture_results.get("mc_probs")
    podos_probs = fixture_results.get("podos_probs") # Placeholder for Podos
    lambdas = fixture_results.get("lambdas", (None, None))
    lambda_h_str = f"{lambdas[0]:.3f}" if lambdas[0] is not None else "N/A"
    lambda_a_str = f"{lambdas[1]:.3f}" if lambdas[1] is not None else "N/A"

    mc_h = mc_probs.get('prob_H', 0) if mc_probs else 0
    mc_d = mc_probs.get('prob_D', 0) if mc_probs else 0
    mc_a = mc_probs.get('prob_A', 0) if mc_probs else 0

    # Use placeholder data for Podos if not available
    if not podos_probs:
        logger.warning("Podos probabilities not found in fixture_results. Using placeholders.")
        # Make placeholder visually distinct but summing to 1
        podos_h_ph, podos_d_ph, podos_a_ph = 0.33, 0.34, 0.33
        podos_probs = {'prob_H': podos_h_ph, 'prob_D': podos_d_ph, 'prob_A': podos_a_ph}
    else:
        podos_h_ph = podos_probs.get('prob_H', 0)
        podos_d_ph = podos_probs.get('prob_D', 0)
        podos_a_ph = podos_probs.get('prob_A', 0)

    if not mc_probs: # Only show message if MC probs are missing
        ax.text(0.5, 0.5, "MC Probs\nNot Available", ha='center', va='center', fontsize=10, color='red')
        ax.set_title("Model Comparison: 1X2")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    outcomes = ['Home Win', 'Draw', 'Away Win']
    x = np.arange(len(outcomes))
    width = 0.35

    # MC Colors (Prioritized) - Use slightly more distinct colors
    mc_colors = ['#87CEEB', '#D3D3D3', '#F08080'] # SkyBlue, LightGrey, LightCoral
    # Podos Colors (Placeholder) - Use darker, contrasting colors
    podos_colors = ['#00008B', '#696969', '#8B0000'] # DarkBlue, DimGray, DarkRed


    rects1 = ax.bar(x - width/2, [mc_h, mc_d, mc_a], width, label=f'MC (λ {lambda_h_str}/{lambda_a_str})', color=mc_colors)
    rects2 = ax.bar(x + width/2, [podos_h_ph, podos_d_ph, podos_a_ph], width, label='Podos (Placeholder!)', color=podos_colors)

    ax.set_ylabel('Probability')
    ax.set_title('Model Comparison: 1X2')
    ax.set_xticks(x)
    ax.set_xticklabels(outcomes)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    # Function to add labels
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    add_labels(rects1)
    add_labels(rects2)


# --- Helper: Plot O/U 2.5 and BTTS ---
def _plot_ou_btts_on_ax(ax, fixture_results: Dict[str, Any]):
    """Draws the O/U 2.5 and BTTS probabilities onto a given Axes object."""
    mc_probs = fixture_results.get("mc_probs")

    if not mc_probs:
        ax.text(0.5, 0.5, "MC Probs\nNot Available", ha='center', va='center', fontsize=10, color='red')
        ax.set_title("O/U 2.5 & BTTS")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    o25 = mc_probs.get('prob_O2.5', 0)
    u25 = 1.0 - o25 if o25 is not None else 0 # Handle None case
    btts_yes = mc_probs.get('prob_BTTS Yes', 0)
    btts_no = 1.0 - btts_yes if btts_yes is not None else 0 # Handle None case

    categories = ['O/U 2.5', 'BTTS']
    x = np.arange(len(categories))
    width = 0.35

    # Ensure probabilities are valid numbers
    o25 = o25 if o25 is not None else 0
    u25 = u25 if u25 is not None else 0
    btts_yes = btts_yes if btts_yes is not None else 0
    btts_no = btts_no if btts_no is not None else 0

    # Use clearer color pairs
    over_yes_colors = ['#FF6347', '#3CB371'] # Tomato, MediumSeaGreen
    under_no_colors = ['#ADD8E6', '#FFA07A'] # LightBlue, LightSalmon

    rects1 = ax.bar(x - width/2, [o25, btts_yes], width, label='Over / Yes', color=over_yes_colors)
    rects2 = ax.bar(x + width/2, [u25, btts_no], width, label='Under / No', color=under_no_colors)

    ax.set_ylabel('Probability')
    ax.set_title('MC Probabilities: O/U 2.5 & BTTS')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    # Function to add labels
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    add_labels(rects1)
    add_labels(rects2)


# --- Helper: Plot Goal Distribution (CHANGED TO LEVY PDF) ---
def _plot_goal_pdfs_on_ax(ax, fixture_results: Dict[str, Any], max_goals_axis: int = 8):
    """Draws the Levy Stable goal distribution PDFs onto a given Axes object."""
    lambdas = fixture_results.get("lambdas")
    home_team = fixture_results.get("home_team", "Home")
    away_team = fixture_results.get("away_team", "Away")

    if lambdas is None or lambdas[0] is None or lambdas[1] is None:
        ax.text(0.5, 0.5, "Lambdas (xG)\nNot Available", ha='center', va='center', fontsize=10, color='red')
        ax.set_title("Goal Distribution PDF")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    lambda_home, lambda_away = lambdas
    lambda_home = max(0.01, lambda_home) # Ensure lambdas > 0 for sqrt
    lambda_away = max(0.01, lambda_away)

    # Define Levy Stable parameters (could be made configurable)
    alpha = 1.6  # Stability parameter (tail heaviness)
    beta = 0.0   # Skewness parameter (0 = symmetric)

    # Heuristic for scale based on lambda (expected goals)
    scale_home = 0.5 + 0.25 * np.sqrt(lambda_home)
    scale_away = 0.5 + 0.25 * np.sqrt(lambda_away)

    # Generate x-axis (goals)
    x_goals = np.linspace(-0.5, max_goals_axis + 0.5, 200) # Finer resolution for smooth PDF

    # Calculate PDFs
    try:
        pdf_home = levy_stable.pdf(x_goals, alpha, beta, loc=lambda_home, scale=scale_home)
        pdf_away = levy_stable.pdf(x_goals, alpha, beta, loc=lambda_away, scale=scale_away)
    except Exception as e:
         logger.error(f"Error calculating Levy PDF: {e}", exc_info=True)
         ax.text(0.5, 0.5, "Levy PDF Calculation\nError", ha='center', va='center', fontsize=10, color='red')
         ax.set_title("Goal Distribution PDF")
         return

    # Plotting
    ax.plot(x_goals, pdf_home, label=f'{home_team} (λ={lambda_home:.2f})', color='#1f77b4', linewidth=2) # Blue
    ax.plot(x_goals, pdf_away, label=f'{away_team} (λ={lambda_away:.2f})', color='#d62728', linewidth=2) # Red

    ax.set_xlabel('Number of Goals')
    ax.set_ylabel('Probability Density')
    ax.set_title('Levy Stable Goal Distribution PDF')
    ax.legend(fontsize=8)
    ax.set_xlim(left=-0.5, right=max_goals_axis + 0.5)
    ax.set_ylim(bottom=0) # Ensure y-axis starts at 0
    ax.grid(True, linestyle='--', alpha=0.6) # Add subtle grid

    # Add parameter annotations
    param_text = (f"Levy Params (α={alpha:.1f}, β={beta:.1f})\n"
                  f"Home: loc={lambda_home:.2f}, scale={scale_home:.2f}\n"
                  f"Away: loc={lambda_away:.2f}, scale={scale_away:.2f}")
    ax.annotate(param_text, xy=(0.97, 0.97), xycoords='axes fraction',
                verticalalignment='top', horizontalalignment='right', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))


# --- Helper: Display Top Selections (REMOVED) ---
# def _plot_insights_on_ax(ax, fixture_results: Dict[str, Any]):
#     ...


# --- Main Plotting Function ---
def create_combined_fixture_plot(
    fixture_results: Dict[str, Any],
    output_dir: str,
    max_goals_matrix: int = 5,
    max_goals_pdf_axis: int = 8 # Max goals for PDF plot axis
):
    """
    Generates and saves a single 2x2 plot combining key fixture insights.

    Requires 'lambdas' tuple in fixture_results for the goal distribution plot.
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 9)) # Adjusted figsize for better spacing

    # --- Draw each subplot ---
    mesh = _plot_goal_matrix_on_ax(axes[0, 0], fixture_results, max_goals_matrix)
    if mesh: # Add colorbar if matrix was plotted
        fig.colorbar(mesh, ax=axes[0, 0], label='Probability', fraction=0.046, pad=0.04)

    _plot_mc_vs_podos_on_ax(axes[0, 1], fixture_results)
    _plot_ou_btts_on_ax(axes[1, 0], fixture_results)
    _plot_goal_pdfs_on_ax(axes[1, 1], fixture_results, max_goals_axis=max_goals_pdf_axis) # Use new PDF plot

    # --- Overall Figure Title ---
    lambdas = fixture_results.get("lambdas", (None, None))
    lambda_h_str = f"{lambdas[0]:.3f}" if lambdas[0] is not None else "N/A"
    lambda_a_str = f"{lambdas[1]:.3f}" if lambdas[1] is not None else "N/A"
    fig.suptitle(f'Fixture Analysis: {home_team} vs {away_team} (ID: {fixture_id})\n MC Lambdas (H/A): {lambda_h_str} / {lambda_a_str}', fontsize=14)

    # --- Adjust Layout & Save ---
    try:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent title overlap
        plt.savefig(plot_path)
        logger.info(f"Saved Combined Analysis plot to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to create or save combined plot {plot_path}: {e}", exc_info=True)
    finally:
        # Ensure figure is closed regardless of saving success
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig) 