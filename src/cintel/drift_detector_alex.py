# === DECLARE IMPORTS ===

import logging
from pathlib import Path
from typing import Final

import polars as pl
from datafun_toolkit.logger import get_logger, log_header, log_path

# === CONFIGURE LOGGER ===

LOG: logging.Logger = get_logger("P5", level="DEBUG")

# === DEFINE GLOBAL PATHS ===

ROOT_DIR: Final[Path] = Path.cwd()
DATA_DIR: Final[Path] = ROOT_DIR / "data"
ARTIFACTS_DIR: Final[Path] = ROOT_DIR / "artifacts"

REFERENCE_FILE: Final[Path] = DATA_DIR / "reference_metrics_case.csv"
CURRENT_FILE: Final[Path] = DATA_DIR / "current_metrics_case.csv"

OUTPUT_FILE: Final[Path] = ARTIFACTS_DIR / "drift_summary_alex.csv"
SUMMARY_LONG_FILE: Final[Path] = ARTIFACTS_DIR / "drift_summary_long_alex.csv"

# === DEFINE THRESHOLDS ===

# Analysts need to know their data and
# choose thresholds that make sense for their specific use case.

# Review the reference metrics to understand typical values
# and variability before setting thresholds.

# In this example, we compare current metrics to a reference period
# and flag drift when the difference exceeds these thresholds:

REQUESTS_DRIFT_THRESHOLD: Final[float] = 20.0
ERRORS_DRIFT_THRESHOLD: Final[float] = 2.0
LATENCY_DRIFT_THRESHOLD: Final[float] = 1000.0
# Add Standard Deviation (Sigma) THRESHOLDS
SIGMA_DRIFT_THRESHOLD: Final[float] = 2.0
# Add Rate of Change Thresholds (DELTARATE)
# DELTARATE_DRIFT_THRESHOLD: Final[float] = 0.5
# Rate of change may not be meaningful for this step since we only have one reference and current period

# === DEFINE THE MAIN FUNCTION ===


def main() -> None:
    """Run the pipeline.

    log_header() logs a standard run header.
    log_path() logs repo-relative paths (privacy-safe).
    """
    log_header(LOG, "CINTEL")

    LOG.info("========================")
    LOG.info("START main()")
    LOG.info("========================")

    log_path(LOG, "ROOT_DIR", ROOT_DIR)
    log_path(LOG, "REFERENCE_FILE", REFERENCE_FILE)
    log_path(LOG, "CURRENT_FILE", CURRENT_FILE)
    log_path(LOG, "OUTPUT_FILE", OUTPUT_FILE)

    # Ensure the artifacts folder exists.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path(LOG, "ARTIFACTS_DIR", ARTIFACTS_DIR)

    # ----------------------------------------------------
    # STEP 1: READ REFERENCE AND CURRENT CSV INTO DATAFRAMES
    # ----------------------------------------------------
    reference_df = pl.read_csv(REFERENCE_FILE)
    current_df = pl.read_csv(CURRENT_FILE)

    LOG.info(f"Loaded {reference_df.height} reference records")
    LOG.info(f"Loaded {current_df.height} current records")

    # ----------------------------------------------------
    # STEP 2: CALCULATE METRICS FOR EACH PERIOD
    # ----------------------------------------------------
    # Let's add standard deviation and rate of change calculations to the summary tables.
    # We'll only look at requests since this is small edit task and we can apply the same logic to errors and latency if needed.

    reference_summary_df = reference_df.select(
        [
            pl.col("requests").mean().alias("reference_avg_requests"),
            pl.col("requests").std().alias("reference_sigma_requests"),
            pl.col("errors").mean().alias("reference_avg_errors"),
            pl.col("total_latency_ms").mean().alias("reference_avg_latency_ms"),
        ]
    )

    current_summary_df = current_df.select(
        [
            pl.col("requests").mean().alias("current_avg_requests"),
            pl.col("requests").std().alias("current_sigma_requests"),
            pl.col("errors").mean().alias("current_avg_errors"),
            pl.col("total_latency_ms").mean().alias("current_avg_latency_ms"),
        ]
    )

    # Let's also add a one rolling mean window for requests to see if it's feasible to use in the custom project.
    # We'll use the expression logic but separate the dataframe since the lengths will differ.
    baseline_rolling_df = reference_df.select(
        pl.col("requests")
        .rolling_mean(window_size=3)
        .alias("baseline_rolling_mean_requests")
    )
    current_rolling_df = current_df.select(
        pl.col("requests")
        .rolling_mean(window_size=3)
        .alias("current_rolling_mean_requests")
    )
    rolling_difference_df = baseline_rolling_df - current_rolling_df
    LOG.info(
        f"Calculated rolling mean difference for requests: {rolling_difference_df}"
    )

    # ----------------------------------------------------
    # STEP 3: COMBINE THE TWO ONE-ROW SUMMARY TABLES
    # ----------------------------------------------------
    # Each summary table has one row.
    #
    # reference_summary_df:
    #   reference_avg_requests
    #   reference_avg_errors
    #   reference_avg_latency_ms
    #
    # current_summary_df:
    #   current_avg_requests
    #   current_avg_errors
    #   current_avg_latency_ms
    #
    # We combine them horizontally so both sets of values
    # appear side-by-side in a single row using the
    # concatenate function (pl.concat).
    #
    # This makes it easy to calculate:
    #   current value - reference value

    combined_df: pl.DataFrame = pl.concat(
        [reference_summary_df, current_summary_df],
        how="horizontal",
    )

    # ----------------------------------------------------
    # STEP 4: DEFINE DIFFERENCE RECIPES
    # ----------------------------------------------------
    # A difference recipe calculates:
    #
    #     current average - reference average
    #
    # Positive values mean the current period is larger.
    # Negative values mean the current period is smaller.

    requests_mean_difference_recipe: pl.Expr = (
        (pl.col("current_avg_requests") - pl.col("reference_avg_requests"))
        .round(2)
        .alias("requests_mean_difference")
    )

    requests_sigma_difference_recipe: pl.Expr = (
        (pl.col("current_sigma_requests") - pl.col("reference_sigma_requests"))
        .round(2)
        .alias("requests_sigma_difference")
    )

    errors_mean_difference_recipe: pl.Expr = (
        (pl.col("current_avg_errors") - pl.col("reference_avg_errors"))
        .round(2)
        .alias("errors_mean_difference")
    )

    latency_mean_difference_recipe: pl.Expr = (
        (pl.col("current_avg_latency_ms") - pl.col("reference_avg_latency_ms"))
        .round(2)
        .alias("latency_mean_difference_ms")
    )

    # ----------------------------------------------------
    # STEP 4.1: APPLY THE DIFFERENCE RECIPES TO EXPAND THE DATAFRAME
    # ----------------------------------------------------
    drift_df: pl.DataFrame = combined_df.with_columns(
        [
            requests_mean_difference_recipe,
            requests_sigma_difference_recipe,
            errors_mean_difference_recipe,
            latency_mean_difference_recipe,
        ]
    )

    # ----------------------------------------------------
    # STEP 5: DEFINE DRIFT FLAG RECIPES
    # ----------------------------------------------------
    # A drift flag recipe checks whether the absolute size
    # of the difference exceeds a threshold.
    #
    # We use abs() because either direction may matter:
    # - much higher than reference
    # - much lower than reference

    requests_is_drifting_flag_recipe: pl.Expr = (
        pl.col("requests_mean_difference").abs() > REQUESTS_DRIFT_THRESHOLD
    ).alias("requests_is_drifting_flag")
    requests_sigma_is_drifting_flag_recipe: pl.Expr = (
        pl.col("requests_sigma_difference").abs() > SIGMA_DRIFT_THRESHOLD
    ).alias("requests_sigma_is_drifting_flag")
    errors_is_drifting_flag_recipe: pl.Expr = (
        pl.col("errors_mean_difference").abs() > ERRORS_DRIFT_THRESHOLD
    ).alias("errors_is_drifting_flag")

    latency_is_drifting_flag_recipe: pl.Expr = (
        pl.col("latency_mean_difference_ms").abs() > LATENCY_DRIFT_THRESHOLD
    ).alias("latency_is_drifting_flag")

    # ----------------------------------------------------
    # STEP 5.1: APPLY THE DRIFT FLAG RECIPES TO EXPAND THE DATAFRAME
    # ----------------------------------------------------
    drift_df = drift_df.with_columns(
        [
            requests_is_drifting_flag_recipe,
            requests_sigma_is_drifting_flag_recipe,
            errors_is_drifting_flag_recipe,
            latency_is_drifting_flag_recipe,
        ]
    )

    LOG.info("Calculated summary differences and drift flags")

    # ----------------------------------------------------
    # STEP 6: SAVE THE FLAT DRIFT SUMMARY AS AN ARTIFACT
    # ----------------------------------------------------
    drift_df.write_csv(OUTPUT_FILE)
    LOG.info(f"Wrote drift summary file: {OUTPUT_FILE}")

    # Take a look at the summary dataframe.
    # Lots of columns with one row of values.
    LOG.info("Drift summary dataframe:")
    LOG.info(drift_df)
    LOG.info("Let's make that a bit nicer to read...")
    LOG.info("All remaining steps are about creating a nicer display.")

    # ----------------------------------------------------
    # OPTIONAL STEP 6.1: LOG THE SUMMARY ONE FIELD PER LINE
    # ----------------------------------------------------
    # drift_df has one row with many columns.
    # Convert that one row to a dictionary so we can log:
    # column_name: value

    # The Polars to_dicts() function returns a list of dictionaries, one per row.
    # the [0] gets the first (and only) dictionary from the list.
    # We often count starting at zero
    # because the first row is 0 away from the start of the dataframe.
    drift_summary_dict = drift_df.to_dicts()[0]

    LOG.info("========================")
    LOG.info("Drift Detection Process: ")
    LOG.info("========================")
    LOG.info("1. Summarize each period with means.")
    LOG.info("2. Compute difference of means.")
    LOG.info("3. Flag drift if absolute difference of means exceeds a threshold.")
    LOG.info("========================")

    LOG.info("Drift summary (one field per line):")
    for field_name, field_value in drift_summary_dict.items():
        LOG.info(f"{field_name}: {field_value}")

    # ----------------------------------------------------
    # OPTIONAL STEP 7: CREATE A LONG-FORM ARTIFACT FOR DISPLAY
    # ----------------------------------------------------
    # Create a second artifact with one field per row.
    # This is easier to read than a single very wide row.
    # We create a new dataframe with two columns:
    # - field_name: the name of the summary field
    # - field_value: the value of the summary field (converted to string for display)

    drift_summary_long_df = pl.DataFrame(
        {
            "field_name": list(drift_summary_dict.keys()),
            "field_value": [str(value) for value in drift_summary_dict.values()],
        }
    )
    # ----------------------------------------------------
    # OPTIONAL STEP 7.1: SAVE THE LONG-FORM DRIFT SUMMARY AS AN ARTIFACT
    # ----------------------------------------------------
    drift_summary_long_df.write_csv(SUMMARY_LONG_FILE)
    LOG.info(f"Wrote long summary file: {SUMMARY_LONG_FILE}")

    LOG.info("========================")
    LOG.info("Pipeline executed successfully!")
    LOG.info("========================")
    LOG.info("END main()")


# === CONDITIONAL EXECUTION GUARD ===

if __name__ == "__main__":
    main()
