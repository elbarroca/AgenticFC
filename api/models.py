# models.py (or wherever you define your Pydantic models)

from pydantic import BaseModel, Field, field_validator, ConfigDict, FilePath, DirectoryPath
from typing import List, Optional, Literal, Dict, Any, Union
from decimal import Decimal
import os # Import os

# Import defaults needed for the simplified model
from score_data.paper_generator import (
    DEFAULT_MIN_ODDS, DEFAULT_MIN_PROBABILITY, DEFAULT_EDGE_THRESHOLD,
    DEFAULT_PAPER_SIZES, DEFAULT_MAX_PAPERS_PER_SIZE_GREEDY,
    DEFAULT_FILTER_MIN_COMBINED_ODDS, DEFAULT_FILTER_MAX_COMBINED_ODDS,
    DEFAULT_FILTER_MIN_AVG_EDGE, DEFAULT_RANKING_STRATEGY,
    DEFAULT_PLOT_OUTPUT_DIR, DEFAULT_PLOTS_TO_GENERATE,
    DEFAULT_USE_CVXPY, DEFAULT_PAPER_BUILD_STRATEGY,
    DEFAULT_CVXPY_MAX_COMBINED_ODDS, DEFAULT_CVXPY_MIN_COMBINED_PROB,
    DEFAULT_FILTER_MIN_COMBINED_PROB, DEFAULT_RANKING_STRATEGY,
    DEFAULT_INPUT_FILE, DEFAULT_OUTPUT_FILE, DEFAULT_WEIGHT_PROB,
    DEFAULT_WEIGHT_EDGE, DEFAULT_WEIGHT_VALUE_RATIO,
    DEFAULT_TOP_N_PER_GAME, DEFAULT_MAX_ODDS,
    DEFAULT_FILTER_LEAGUES, DEFAULT_FILTER_TEAMS,
    DEFAULT_ENABLE_PLOTTING
)

# Helper function to resolve relative paths if needed by API defaults
def resolve_default_path(relative_path: str, base_dir: str = ".") -> str:
    # In a real API, you might want base_dir configurable
    # For now, assume paths are relative to project root or handled by main.py
    return relative_path # Keep it simple for now, main.py handles resolution

class EfficiencyWeights(BaseModel):
    """Defines weights for calculating efficiency score."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    weight_prob: Decimal = Field(default=DEFAULT_WEIGHT_PROB, ge=0.0, le=1.0)
    weight_edge: Decimal = Field(default=DEFAULT_WEIGHT_EDGE, ge=0.0, le=1.0)
    weight_value_ratio: Decimal = Field(default=DEFAULT_WEIGHT_VALUE_RATIO, ge=0.0, le=1.0)

    @field_validator('weight_prob', 'weight_edge', 'weight_value_ratio')
    @classmethod
    def check_decimal(cls, v):
        if isinstance(v, (int, float)):
             return Decimal(str(v))
        if not isinstance(v, Decimal):
             raise ValueError("Must be a Decimal or convertible to Decimal")
        return v

class PlotConfig(BaseModel):
    """Configuration for a single plot."""
    filename: str
    x: str
    y: str
    color: Optional[str] = None
    size: Optional[str] = None
    title: Optional[str] = None
    x_log: bool = False
    y_log: bool = False

class PaperGenerationRequest(BaseModel):
    """
    Comprehensive Pydantic model for requesting optimized betting papers,
    aligning with score_data.paper_generator.generate_papers parameters.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='ignore')

    # --- Input/Output Control ---
    # input_file will be handled by API logic based on filename + base dir
    input_filename: str = Field(
        description="Filename of the batch prediction results JSON file within the API's configured input directory."
    )
    # output_file and plot_output_dir are managed by API using request_id
    debug: bool = Field(default=False, description="Enable detailed debug logging.")

    # --- Selection Filtering ---
    min_edge: Decimal = Field(
        default=DEFAULT_EDGE_THRESHOLD,
        description="Minimum edge required for a selection."
    )
    min_probability: Decimal = Field(
        default=DEFAULT_MIN_PROBABILITY, ge=Decimal('0.0'), le=Decimal('1.0'),
        description="Minimum probability for a selection."
    )
    min_odds: Optional[Decimal] = Field(
        default=DEFAULT_MIN_ODDS, gt=Decimal('1.0'),
        description="Minimum odds for *individual selections* (optional)."
    )
    max_odds: Optional[Decimal] = Field(
        default=DEFAULT_MAX_ODDS, gt=Decimal('1.0'),
        description="Maximum odds for *individual selections* (optional)."
    )
    top_n_per_game: int = Field(
        default=DEFAULT_TOP_N_PER_GAME, gt=0,
        description="Maximum selections to consider per game after initial filtering."
    )

    # --- Efficiency Score ---
    # Use nested model for clarity
    efficiency_weights: EfficiencyWeights = Field(
        default_factory=EfficiencyWeights, # Use default factory for nested model
        description="Weights for calculating the efficiency score."
    )

    # --- Paper Generation ---
    paper_sizes: List[int] = Field(
        default_factory=lambda: list(DEFAULT_PAPER_SIZES), # Default: [3]
        description="List of desired number of selections (legs) per paper."
    )
    paper_build_strategy: Literal['efficiency', 'highest_edge', 'highest_probability'] = Field(
        default=DEFAULT_PAPER_BUILD_STRATEGY,
        description="Strategy for greedy paper building."
    )
    use_cvxpy: bool = Field(
        default=DEFAULT_USE_CVXPY,
        description="Use CVXPY optimization instead of the greedy strategy."
    )
    # Note: max_papers logic depends on use_cvxpy, handled in API if needed,
    # but we can add both defaults from generator here for completeness.
    max_papers_per_size_greedy: int = Field(
         default=DEFAULT_MAX_PAPERS_PER_SIZE_GREEDY, gt=0,
         description="Max papers per size for the 'greedy' strategy."
     )
    # CVXPY finds only one optimal paper per size, so its max is effectively 1.

    # --- CVXPY Constraints (Only used if use_cvxpy is True) ---
    cvxpy_max_combined_odds: Optional[Decimal] = Field(
        default=DEFAULT_CVXPY_MAX_COMBINED_ODDS, gt=Decimal('1.0'),
        description="Max combined odds constraint for CVXPY optimization (optional)."
    )
    cvxpy_min_combined_prob: Optional[Decimal] = Field(
        default=DEFAULT_CVXPY_MIN_COMBINED_PROB, ge=Decimal('0.0'), le=Decimal('1.0'),
        description="Min combined probability constraint for CVXPY optimization (optional)."
    )

    # --- Final Paper Filtering (Post-Generation) ---
    filter_leagues: Optional[List[str]] = Field(
        default=DEFAULT_FILTER_LEAGUES,
        description="Filter papers to only include these specific leagues (case-insensitive, optional)."
    )
    filter_teams: Optional[List[str]] = Field(
        default=DEFAULT_FILTER_TEAMS,
        description="Filter papers to ensure they include at least one of these teams (case-insensitive, optional)."
    )
    filter_min_combined_odds: Optional[Decimal] = Field(
        default=DEFAULT_FILTER_MIN_COMBINED_ODDS, gt=Decimal('1.0'),
        description="Minimum *combined odds* required for final papers (optional)."
    )
    filter_max_combined_odds: Optional[Decimal] = Field(
        default=DEFAULT_FILTER_MAX_COMBINED_ODDS, gt=Decimal('1.0'),
        description="Maximum *combined odds* allowed for final papers (optional)."
    )
    filter_min_combined_prob: Optional[Decimal] = Field(
        default=DEFAULT_FILTER_MIN_COMBINED_PROB, ge=Decimal('0.0'), le=Decimal('1.0'),
        description="Minimum *combined probability* required for final papers (optional)."
    )
    filter_min_avg_edge: Optional[Decimal] = Field(
        default=DEFAULT_FILTER_MIN_AVG_EDGE,
        description="Minimum *average edge* required for final papers (optional)."
    )

    # --- Ranking ---
    ranking_strategy: Literal['combined_prob_then_edge', 'avg_efficiency_score'] = Field(
        default=DEFAULT_RANKING_STRATEGY,
        description="How to rank the resulting papers."
    )

    # --- Plotting ---
    enable_plotting: bool = Field(
        default=DEFAULT_ENABLE_PLOTTING,
        description="Generate plots for the results."
    )
    # plot_output_dir is managed by API using request_id
    plots_to_generate: List[PlotConfig] = Field(
        default_factory=lambda: list(DEFAULT_PLOTS_TO_GENERATE), # Use default factory
        description="Configuration for plots to generate if plotting is enabled."
    )

    # --- Field Validators ---
    @field_validator('min_edge', 'min_probability', 'min_odds', 'max_odds',
                     'cvxpy_max_combined_odds', 'cvxpy_min_combined_prob',
                     'filter_min_combined_odds', 'filter_max_combined_odds',
                     'filter_min_combined_prob', 'filter_min_avg_edge')
    @classmethod
    def check_optional_decimal(cls, v: Optional[Union[Decimal, int, float, str]]) -> Optional[Decimal]:
        if v is None:
            return None
        try:
            # Allow conversion from int/float/str
            return Decimal(str(v))
        except Exception as e:
             raise ValueError(f"Invalid Decimal value: {v}") from e

    @field_validator('paper_sizes')
    @classmethod
    def check_paper_sizes_positive(cls, sizes: List[int]) -> List[int]:
        if not isinstance(sizes, list):
             raise ValueError("paper_sizes must be a list.")
        valid_sizes = [s for s in sizes if isinstance(s, int) and s > 0]
        if not valid_sizes:
            # Fallback to default if input list is empty or contains only invalid sizes
             return list(DEFAULT_PAPER_SIZES)
        # Return unique, sorted list of valid sizes
        return sorted(list(set(valid_sizes)))

    @field_validator('filter_leagues', 'filter_teams')
    @classmethod
    def check_string_list(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError("Must be a list of strings or null.")
        # Return unique list, preserving case from input
        return list(dict.fromkeys(value))


# --- Response Model ---

class GenerationSummary(BaseModel):
    """Summary metrics of the generation process."""
    total_matches_input: int
    unique_fixtures_processed: int
    skipped_duplicate_fixtures: int
    fixtures_with_valid_selections: int
    total_selections_considered: int
    papers_generated_before_filtering: int
    papers_remaining_after_filtering: int
    plots_generated: int

class GenerationInfo(BaseModel):
    """Metadata about the paper generation run."""
    request_id: str # Added by the API
    generated_at: str
    execution_duration_seconds: float
    input_file_processed: Optional[str] = None # Basename added by API
    output_file_generated: Optional[str] = None # Basename added by API
    plotting_enabled: bool
    plot_output_dir: Optional[str] = None # Basename added by API
    generated_plot_files: List[str]
    settings: Dict[str, Any] # Contains the processed settings used
    summary: GenerationSummary
    status: str
    error_message: Optional[str] = None
    # URLs/Paths added by API
    output_file_url: Optional[str] = None
    output_file_relative_path: Optional[str] = None
    generated_plot_urls: Optional[List[str]] = None
    generated_plot_relative_paths: Optional[List[str]] = None


class PaperGenerationResponse(BaseModel):
    """
    Defines the structure of the response returned by the /generate-papers/ endpoint.
    """
    generation_info: GenerationInfo
    ranked_filtered_papers: List[Dict[str, Any]] # List of paper dicts (structure defined in paper_generator)
    