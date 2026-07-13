#!/usr/bin/env Rscript
# BGLR backend for BayesA/BayesB/BayesCpi
# Args:
#   1: X.csv (no header)
#   2: y.csv (no header)
#   3: method (BayesA | BayesB | BayesCpi)
#   4: output b.csv
#   5: output mu.csv
#   6: n_iter
#   7: burn_in
#   8: seed

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 8) {
stop("Usage: 02g_backend_bglr.R <X.csv> <y.csv> <method> <b.csv> <mu.csv> <n_iter> <burn_in> <seed>")
}

X_path <- args[[1]]
y_path <- args[[2]]
method <- args[[3]]
b_path <- args[[4]]
mu_path <- args[[5]]
n_iter <- as.integer(args[[6]])
burn_in <- as.integer(args[[7]])
seed <- as.integer(args[[8]])

suppressMessages(library(BGLR))

# #region agent log
debug_log <- function(hypothesis_id, location, message, data_json) {
  # Optional debug trace. Only writes when BREEDAI_DEBUG_LOG points at a real file;
  # wrapped in try() so debug logging can never break the actual BGLR run.
  target <- Sys.getenv("BREEDAI_DEBUG_LOG", unset = "")
  if (!nzchar(target)) return(invisible(NULL))
  ts <- as.integer(as.numeric(Sys.time()) * 1000)
  line <- sprintf('{"sessionId":"debug-session","runId":"post-fix","hypothesisId":"%s","location":"%s","message":"%s","data":%s,"timestamp":%d}\n',
                  hypothesis_id, location, message, data_json, ts)
  try(cat(line, file = target, append = TRUE), silent = TRUE)
}
# #endregion agent log

X <- as.matrix(read.csv(X_path, header = FALSE, check.names = FALSE))
y <- as.numeric(read.csv(y_path, header = FALSE)[, 1])

# #region agent log
debug_log(
  "H8",
  "02g_backend_bglr.R:34",
  "BGLR backend inputs",
  sprintf('{"method":"%s","X_rows":%d,"X_cols":%d,"y_len":%d,"bglr_version":"%s"}',
          method, nrow(X), ncol(X), length(y), as.character(packageVersion("BGLR")))
)
# #endregion agent log

set.seed(seed)

# BGLR writes scratch files (ETA_*.dat, mu.dat, varE.dat). Direct them to the unique
# per-run temp dir (dirname of the output path) so parallel array tasks don't collide
# on shared filenames and the current directory need not be writable.
saveAt <- file.path(dirname(b_path), "bglr_")

ETA <- list(list(X = X, model = method))
fit <- NULL
fit <- tryCatch(
  {
    BGLR(y = y, ETA = ETA, nIter = n_iter, burnIn = burn_in, verbose = FALSE, saveAt = saveAt)
  },
  error = function(e) {
    if (identical(method, "BayesCpi")) {
      # #region agent log
      debug_log(
        "H9",
        "02g_backend_bglr.R:49",
        "BayesCpi not implemented, fallback to BayesC",
        sprintf('{"error":"%s"}', gsub('"', "'", conditionMessage(e)))
      )
      # #endregion agent log
      ETA2 <- list(list(X = X, model = "BayesC", probIn = 0.5))
      return(BGLR(y = y, ETA = ETA2, nIter = n_iter, burnIn = burn_in, verbose = FALSE, saveAt = saveAt))
    }
    stop(e)
  }
)

write.table(fit$ETA[[1]]$b, b_path, sep = ",", row.names = FALSE, col.names = FALSE)
write.table(fit$mu, mu_path, sep = ",", row.names = FALSE, col.names = FALSE)
