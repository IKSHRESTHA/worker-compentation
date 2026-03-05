# Automated bootstrap for ML project dependencies and environment variables
required_vars <- c("DATA_DIR", "MODEL_DIR", "SECRETS_FILE")

message("Ensuring renv is installed...")
if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv")
}

message("Restoring packages from renv.lock...")
renv::restore(prompt = FALSE)

message("Collecting environment variable values...")
get_var_value <- function(var_name) {
  val <- Sys.getenv(var_name, unset = "")
  if (nzchar(val)) {
    return(val)
  }
  if (!interactive()) {
    stop(sprintf("Environment variable %s not set; rerun script in interactive mode or export it beforehand.", var_name))
  }
  prompt <- sprintf("Enter value for %s: ", var_name)
  val <- readline(prompt)
  if (!nzchar(val)) {
    stop(sprintf("No value provided for %s.", var_name))
  }
  val
}

resolved_vars <- vapply(required_vars, get_var_value, character(1), USE.NAMES = TRUE)

env_path <- file.path(getwd(), ".Renviron")
existing <- if (file.exists(env_path)) readLines(env_path, warn = FALSE) else character()

strip_var <- function(lines, var_name) {
  pattern <- sprintf("^%s\\s*=", var_name)
  lines[!grepl(pattern, lines)]
}

for (var_name in names(resolved_vars)) {
  existing <- strip_var(existing, var_name)
  line <- sprintf("%s=%s", var_name, resolved_vars[[var_name]])
  existing <- c(existing, line)
}

writeLines(existing, env_path)
message("Updated .Renviron with required variables.")

message("Bootstrap complete. Restart your R session to pick up restored libraries and environment variables.")
