# load_data.R
# Loads the raw workers' compensation dataset from data/raw/data.csv
# Uses readr for fast parsing and here for project-relative paths.

library(here)
library(readr)

DATA_PATH <- here("data", "raw", "data.csv")

col_spec <- cols(
  ClaimNumber              = col_character(),
  DateTimeOfAccident       = col_datetime(format = "%Y-%m-%dT%H:%M:%SZ"),
  DateReported             = col_datetime(format = "%Y-%m-%dT%H:%M:%SZ"),
  Age                      = col_integer(),
  Gender                   = col_factor(levels = c("M", "F")),
  MaritalStatus            = col_factor(levels = c("M", "S", "U", "D", "W")),
  DependentChildren        = col_integer(),
  DependentsOther          = col_integer(),
  WeeklyWages              = col_double(),
  PartTimeFullTime         = col_factor(levels = c("F", "P")),
  HoursWorkedPerWeek       = col_double(),
  DaysWorkedPerWeek        = col_double(),
  ClaimDescription         = col_character(),
  InitialIncurredCalimsCost = col_double(),
  UltimateIncurredClaimCost = col_double()
)

claims <- read_csv(DATA_PATH, col_types = col_spec)

View(claims)
