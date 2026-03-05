# Config Bootstrap Guide

This folder tracks every project-wide configuration reference so the ML workflows stay reproducible.

## Dependencies
- R 4.3+ (install from https://cran.r-project.org)
- `renv` for isolating package versions
- Core modeling libraries declared in `../renv.lock`: `tidyverse`, `xgboost`, `shapviz`

## Bootstrap Steps
1. Install the base dependencies above.
2. From the project root, open an R session and run:
   ```r
   install.packages("renv")
   renv::restore()
   ```
   These commands recreate the library pinned in `renv.lock`.
3. Restart the R session so that the restored packages are picked up by RStudio/VS Code.
4. Optional: Run `renv::status()` to confirm the library matches the lockfile.

## Environment Variables
Add the following to your `.Renviron` or shell profile.
- `DATA_DIR` - absolute path to `data/processed` for feature-ready datasets.
- `MODEL_DIR` - absolute path to `models` where trained objects are serialized.
- `SECRETS_FILE` - path to a YAML/JSON credential file (never commit the real file).

Load them in R with `Sys.getenv("VAR_NAME")` and fail fast if any are unset.

## Updating Config
- Add new package requirements via `renv::install("package")` and snapshot with `renv::snapshot()`.
- Document new environment variables or config templates here so the whole team can follow the same playbook.
