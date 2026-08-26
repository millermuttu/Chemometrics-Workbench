# Reference values from R's mdatools, for the quantities scikit-learn does not
# report at all: the Hotelling T2 and SPE limits, and SIMPLS coefficients.
#
# Why R is worth the trouble (#24): chemotools is another Python implementation
# on the same NumPy, so agreeing with it says less than agreeing with a
# different language and a different lineage. mdatools is Sergey Kucheryavskiy's
# implementation of the Chemometrics conventions, and its PLS is SIMPLS where
# ours is NIPALS - which pls-regression.md section 2 records as coinciding for a
# single response in coefficients and predictions, and not in weights.
#
#   Rscript tests/fixtures/r_mdatools_reference.R <matrix-dir> <output.json>
#
# The matrices come from export_for_r.py and are already centred, so every fit
# below passes center = FALSE: mdatools centres by default, and centring twice
# would make the comparison meaningless rather than merely wrong.

suppressMessages(library(mdatools))
suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
indir <- if (length(args) > 0) args[1] else "build/r-reference"
outfile <- if (length(args) > 1) args[2] else "tests/fixtures/r_mdatools_values.json"

N_COMPONENTS <- 5
ALPHA <- 0.05
DATASETS <- c("corn", "gasoline", "tecator")

read_matrix <- function(path) as.matrix(read.table(path, sep = "\t", header = FALSE))

results <- list()

for (name in DATASETS) {
  x <- read_matrix(file.path(indir, paste0(name, ".x.tsv")))
  y <- as.numeric(read_matrix(file.path(indir, paste0(name, ".y.tsv"))))

  # lim.type = "jm" is Jackson-Mudholkar, which is the formula pca.md section 8
  # specifies. mdatools defaults to "ddmoments", a data-driven limit that is a
  # different statistic rather than a different rounding - so it is named here
  # explicitly rather than left to the default.
  model <- pca(x, ncomp = N_COMPONENTS, center = FALSE, scale = FALSE,
               lim.type = "jm", alpha = ALPHA)

  spe_limit <- unname(model$Qlim["Extremes limits", N_COMPONENTS])
  t2_limit <- unname(model$T2lim["Extremes limits", N_COMPONENTS])

  # SIMPLS. Valid against our NIPALS for coefficients and predictions only.
  pls_model <- pls(x, y, ncomp = N_COMPONENTS, center = FALSE, scale = FALSE,
                   method = "simpls", cv = NULL)
  coefficients <- as.numeric(pls_model$coeffs$values[, N_COMPONENTS, 1])

  results[[name]] <- list(
    spe_limit = spe_limit,
    hotelling_t2_limit = t2_limit,
    pls_coefficients = coefficients,
    n_samples = nrow(x),
    n_variables = ncol(x)
  )
}

output <- list(
  generated_by = "tests/fixtures/r_mdatools_reference.R",
  r_version = R.version.string,
  mdatools_version = as.character(packageVersion("mdatools")),
  n_components = N_COMPONENTS,
  alpha = ALPHA,
  lim_type = "jm",
  pls_method = "simpls",
  centring = "matrices are pre-centred by export_for_r.py; every fit passes center = FALSE",
  datasets = results
)

write(toJSON(output, auto_unbox = TRUE, digits = 17, pretty = TRUE), outfile)
cat("wrote", outfile, "\n")
