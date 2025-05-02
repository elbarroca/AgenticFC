# main_api.py
import os
import logging
import uuid
from fastapi import FastAPI, HTTPException, Request, Query
from typing import List, Optional # Ensure List and Optional are imported
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware # Optional: For frontend interaction
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool # For running blocking code
import cvxpy as cp # Import cvxpy

# Import Updated Pydantic models and the core logic function
from api.models import PaperGenerationRequest, PaperGenerationResponse # <--- Updated model import
from score_data.paper_generator import generate_papers, logger

# --- FastAPI App Setup ---
app = FastAPI(
    title="Betting Paper Generator API",
    description="API to generate optimized betting papers using greedy or CVXPY logic.",
    version="1.0.0"
)

# --- Optional: CORS Middleware ---
# (Keep or remove as needed)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], # Restrict in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# --- Configure Base Directories ---
BASE_INPUT_DIR = os.path.abspath(os.getenv("PAPER_GENERATOR_INPUT_DIR", os.path.join("data", "output"))) # Adjusted default path
BASE_OUTPUT_DIR = os.path.abspath(os.getenv("PAPER_GENERATOR_OUTPUT_DIR", "static_output"))

os.makedirs(BASE_INPUT_DIR, exist_ok=True)
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
logger.info(f"API Input will be read relative to: {BASE_INPUT_DIR}")
logger.info(f"API Output will be saved relative to: {BASE_OUTPUT_DIR}")

# --- Mount Static Directory ---
app.mount("/output", StaticFiles(directory=BASE_OUTPUT_DIR), name="output")

# --- API Endpoint Definition ---

# Use the new Response model for documentation, but return JSONResponse for flexibility
@app.post("/generate-papers/", response_model=PaperGenerationResponse, response_class=JSONResponse)
async def run_paper_generation(
    request_params: PaperGenerationRequest, # Still accept the full body for advanced use
    request: Request,
    # --- Add Query Parameters for common filters ---
    filter_team: Optional[List[str]] = Query(None, description="Override filter_teams: Apply filter for papers including AT LEAST ONE of these teams (case-insensitive)."),
    filter_league: Optional[List[str]] = Query(None, description="Override filter_leagues: Apply filter for papers including ONLY selections from these leagues (case-insensitive).")
    # Add more query parameters here for other simple overrides if desired (e.g., min_combined_odds)
    # filter_min_combined_odds_override: Optional[float] = Query(None, alias="min_combined_odds", description="Override minimum combined odds.")
):
    """
    Generate betting papers based on input parameters from JSON body.

    Allows overriding `filter_teams` and `filter_leagues` via query parameters
    for simpler filtering requests. Query parameters take precedence over
    values provided in the JSON body for those specific fields.

    Accepts input filename and comprehensive generation/filtering options
    matching the `score_data.paper_generator.generate_papers` function.

    Returns a JSON object containing generation info (including request_id),
    summary, and the ranked list of filtered papers. If plotting is enabled,
    it also includes relative paths/URLs to the generated plot files,
    organized by request ID.
    """
    request_id = str(uuid.uuid4())
    # Log query params if they exist
    query_log = {}
    if filter_team: query_log["filter_team (query)"] = filter_team
    if filter_league: query_log["filter_league (query)"] = filter_league
    logger.info(f"Received request for paper generation (ID: {request_id}). Input filename: {request_params.input_filename}. Query overrides: {query_log or 'None'}")

    try:
        # --- Prepare Parameters for generate_papers ---

        # 1. Construct Absolute Input Path
        input_filename = request_params.input_filename
        if ".." in input_filename or input_filename.startswith("/") or input_filename.startswith("\\"):
             raise ValueError(f"Invalid input_filename: '{input_filename}'. Must be a simple filename relative to the base input directory.")
        input_filepath_abs = os.path.join(BASE_INPUT_DIR, input_filename)

        if not os.path.isfile(input_filepath_abs): # Check if it's a file
            logger.error(f"Input file not found or is not a file at resolved path: {input_filepath_abs}")
            raise HTTPException(status_code=404, detail=f"Input file '{input_filename}' not found or is invalid.")

        # 2. Construct Absolute Output Paths (Organized by Request ID)
        request_output_dir = os.path.join(BASE_OUTPUT_DIR, request_id)
        output_filepath_abs = os.path.join(request_output_dir, "optimized_betting_papers.json")
        plot_dir_abs = os.path.join(request_output_dir, "plots")

        os.makedirs(request_output_dir, exist_ok=True)
        if request_params.enable_plotting:
             os.makedirs(plot_dir_abs, exist_ok=True)

        # 3. Build dictionary using model_dump, excluding the filename field
        # The model ensures all necessary fields/defaults are present.
        try:
            # Use exclude={'input_filename'} to prevent passing it directly
            # We handle paths separately. Keep exclude_unset=True if you want to rely on defaults in generate_papers
            # when a field isn't provided in the request. Use exclude_none=True generally.
            params_for_generator = request_params.model_dump(
                exclude={'input_filename'}, # Exclude the filename itself
                exclude_unset=False, # Include defaults set by the model
                exclude_none=True # Exclude fields explicitly set to None
            )
        except AttributeError: # Fallback for Pydantic v1
            params_for_generator = request_params.dict(
                exclude={'input_filename'},
                exclude_unset=False,
                exclude_none=True
            )

        # --- Apply Query Parameter Overrides ---
        # Query parameters take precedence over the request body values
        if filter_team is not None: # Check for None, empty list is valid input from query
            logger.info(f"Request ID {request_id}: Overriding filter_teams with query parameter: {filter_team}")
            params_for_generator['filter_teams'] = filter_team
        if filter_league is not None:
            logger.info(f"Request ID {request_id}: Overriding filter_leagues with query parameter: {filter_league}")
            params_for_generator['filter_leagues'] = filter_league
        # Add logic for other query param overrides here
        # if filter_min_combined_odds_override is not None:
        #    logger.info(f"Request ID {request_id}: Overriding filter_min_combined_odds with query parameter: {filter_min_combined_odds_override}")
        #    params_for_generator['filter_min_combined_odds'] = Decimal(str(filter_min_combined_odds_override))

        # --- Handle Nested Efficiency Weights (Simplified) ---
        # Pydantic v2 model_dump usually handles nested models correctly.
        # If efficiency_weights is in params_for_generator, it should be a dict.
        # If using Pydantic v1's dict(), nested models might need explicit conversion
        # if efficiency_weights:
        #     if isinstance(params_for_generator['efficiency_weights'], BaseModel):
        #          params_for_generator['efficiency_weights'] = params_for_generator['efficiency_weights'].dict()

        # --- Handle Nested Plot Config (Simplified) ---
        # Similar check if needed for Pydantic v1
        # if 'plots_to_generate' in params_for_generator:
        #     params_for_generator['plots_to_generate'] = [
        #         plot.dict() if isinstance(plot, BaseModel) else plot
        #         for plot in params_for_generator['plots_to_generate']
        #     ]

        # 4. Override/Add necessary paths to the parameters dict
        params_for_generator['input_file'] = input_filepath_abs
        params_for_generator['output_file'] = output_filepath_abs
        params_for_generator['plot_output_dir'] = plot_dir_abs
        # No need to pop 'input_filename' as it was excluded during dump

        logger.debug(f"Processing Request ID {request_id} with parameters prepared for generator (after overrides): {params_for_generator}")

        # --- Execute the core logic in a thread pool ---
        generation_result = await run_in_threadpool(generate_papers, params_for_generator)

        # --- Post-process results for API response ---
        if not generation_result or "generation_info" not in generation_result:
             logger.error(f"Request ID {request_id}: Paper generation function returned unexpected or empty result.")
             raise HTTPException(status_code=500, detail="Internal server error: Paper generation failed unexpectedly.")

        status = generation_result["generation_info"].get("status", "unknown")
        logger.info(f"Request ID {request_id}: Paper generation finished with status: {status}")

        # Add request_id
        generation_result["generation_info"]["request_id"] = request_id

        # Add URLs/relative paths
        base_url = str(request.base_url).rstrip('/')
        output_url_base = f"{base_url}/output/"

        # JSON output file
        json_rel_path = os.path.relpath(output_filepath_abs, BASE_OUTPUT_DIR).replace("\\", "/")
        generation_result["generation_info"]["output_file_url"] = f"{output_url_base}{json_rel_path}"
        generation_result["generation_info"]["output_file_relative_path"] = json_rel_path
        generation_result["generation_info"]["output_file_generated"] = os.path.basename(output_filepath_abs) # Add basename

        # Plot files
        plot_files = generation_result["generation_info"].get("generated_plot_files", [])
        plot_urls = []
        plot_relative_paths = []
        if plot_files and params_for_generator.get('enable_plotting', False):
             plot_dir_basename = os.path.basename(plot_dir_abs) # e.g., "plots"
             request_id_output_rel_path = os.path.relpath(request_output_dir, BASE_OUTPUT_DIR).replace("\\", "/") # e.g., <uuid>
             plot_dir_rel_path = os.path.join(request_id_output_rel_path, plot_dir_basename).replace("\\", "/") # e.g., <uuid>/plots

             generation_result["generation_info"]["plot_output_dir"] = plot_dir_rel_path # Store relative path

             for plot_file_basename in plot_files:
                 # Construct relative path from BASE_OUTPUT_DIR
                 rel_plot_path = os.path.join(plot_dir_rel_path, plot_file_basename).replace("\\", "/") # e.g., <uuid>/plots/plot.png
                 plot_urls.append(f"{output_url_base}{rel_plot_path}")
                 plot_relative_paths.append(rel_plot_path)

        generation_result["generation_info"]["generated_plot_urls"] = plot_urls
        generation_result["generation_info"]["generated_plot_relative_paths"] = plot_relative_paths

        # Add input file basename to info
        generation_result["generation_info"]["input_file_processed"] = os.path.basename(input_filename)

        # Clean up absolute paths from the 'settings' dict within the response for security/tidiness
        # The generator function puts the absolute paths it used here.
        if "settings" in generation_result["generation_info"]:
            generation_result["generation_info"]["settings"].pop("input_file", None)
            generation_result["generation_info"]["settings"].pop("output_file", None)
            generation_result["generation_info"]["settings"].pop("plot_output_dir", None)

        # Determine HTTP status code
        http_status_code = 200
        if "error" in status:
             http_status_code = 500
             logger.error(f"Request ID {request_id}: Error during paper generation: {generation_result['generation_info'].get('error_message', 'Unknown error')}")
        # Consider 200 OK even if no papers are found, as the process completed.
        # elif status.startswith("completed_no_papers") or status.startswith("completed_no_valid"):
             # http_status_code = 200 # Keep as 200

        # Validate the final structure against the response model (optional, FastAPI does this if response_model is used correctly)
        # try:
        #     _ = PaperGenerationResponse(**generation_result)
        # except ValidationError as e:
        #     logger.error(f"Request ID {request_id}: Final response structure failed validation: {e}", exc_info=True)
        #     raise HTTPException(status_code=500, detail="Internal server error: Failed to construct valid response.")

        return JSONResponse(content=generation_result, status_code=http_status_code)

    except ValueError as ve:
        logger.warning(f"Request ID {request_id}: Value Error during request processing: {ve}", exc_info=False) # Don't need full stack for user input errors
        raise HTTPException(status_code=400, detail=f"Invalid parameter value: {ve}")
    except cp.SolverError as se:
        logger.error(f"Request ID {request_id}: CVXPY Solver Error during optimization: {se}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization solver error: {se}. Ensure a suitable MIP solver (like CBC or GLPK_MI) is installed and accessible.")
    except HTTPException as http_exc:
        raise http_exc # Re-raise FastAPI/validation errors
    except Exception as e:
        logger.error(f"Request ID {request_id}: Unhandled exception during paper generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: An unexpected error occurred.")

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# --- Running the API ---
# Command: uvicorn api.main:app --reload --port 8000