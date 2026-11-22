#!/usr/bin/env bash
# Regenerate every table, figure and audit in the manuscript from Data.csv.
# Usage: MELANOMA_DATA=/path/to/Data.csv bash reproduce_all.sh
# Runtime on one CPU core: roughly 1.5-2 hours (each step checkpoints; re-run to resume).
set -euo pipefail
cd "$(dirname "$0")"
for f in 0 1 2 3 4; do python3 step.py fold $f; done
python3 step.py discovery
for i in 0 1 2 3; do python3 step.py loso $i; done
python3 step.py de
python3 step.py singlegene
python3 step.py null
python3 step.py collect
python3 extra_analyses.py composition
python3 extra_analyses.py network
python3 extra_analyses.py roc
python3 null_dist.py
python3 review_checks.py part1
python3 review_checks.py part2
python3 review_checks.py part3
for f in 0 1 2 3 4; do python3 sensitivity.py fold $f; python3 sensitivity.py mlpall $f; done
python3 sensitivity.py discovery
python3 sensitivity.py collect
python3 revision_analyses.py stats
python3 revision_analyses.py classifiers
python3 revision_analyses.py batch
python3 revision_analyses.py dups
python3 mcpcounter_adjust.py
for seed in 7 2026; do for f in 0 1 2 3 4; do MELANOMA_CV_SEED=$seed python3 step.py fold $f; done; done
python3 repeated_cv_summary.py
for f in 0 1 2 3 4; do MELANOMA_GROUPED=1 python3 step.py fold $f; done
python3 tcga_validate_cbio.py
python3 figures.py fig1 fig2 fig3 fig4 fig5 fig6 fig7 fig8 fig9 figS1 figS2 figS3
python3 audit_numbers.py files
python3 audit_numbers.py data
echo "Done. Compare results/audit/*.csv: every row should read PASS."
