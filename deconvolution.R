# Reference-based composition estimates for the merged matrix (xCell, MCP-counter).
# install.packages("remotes"); remotes::install_github("dviraran/xCell")
# remotes::install_github("ebecht/MCPcounter", ref = "master", subdir = "Source")
# Run:  Rscript deconvolution.R path/to/Data.csv
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1], row.names = 1, check.names = FALSE)
d <- d[rownames(d) != "Class", ]
m <- as.matrix(d); storage.mode(m) <- "numeric"
dir.create("results/revision", showWarnings = FALSE, recursive = TRUE)
library(xCell)
xc <- xCellAnalysis(m)
write.csv(t(xc), "results/revision/xcell_scores.csv")
library(MCPcounter)
mcp <- MCPcounter.estimate(m, featuresType = "HUGO_symbols")
write.csv(t(mcp), "results/revision/mcpcounter_scores.csv")
cat("Wrote xcell_scores.csv and mcpcounter_scores.csv; now run: python3 deconv_adjust.py\n")
