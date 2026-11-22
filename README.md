# Benchmarking feature selection for melanoma transcriptomics

Code, data and results for: Pilehvar M, Razmara J, Karimpour J. *Benchmarking feature selection for melanoma
transcriptomics: accuracy, stability, tissue composition and cross-cohort generalisation* (submitted).

`data/Data.csv` is the merged, quantile-normalised matrix (12,339 genes x 321 samples; last row = class:
1 normal skin, 2 primary melanoma, 3 metastatic melanoma) built from GEO series GSE3189, GSE8401, GSE15605 and
GSE46517. Every number in the manuscript is checked by `audit_numbers.py` against `results/`;
`bash reproduce_all.sh` regenerates everything (about 2-3 hours on one CPU core).

# Melanoma biomarker discovery — corrected pipeline

Consensus feature selection across 32 algorithms on merged melanoma
gene-expression data (GSE3189, GSE8401, GSE15605, GSE46517; 321 samples,
12,337 genes after cleaning). Classes: 1 = normal skin (31), 2 = primary
melanoma (153), 3 = metastatic melanoma (137). Class 2 was labelled
"at-risk" in the original thesis, but every class-2 sample is a primary
melanoma in its source series; no benign naevi are in the matrix.

## Install

```
pip install -r requirements.txt
```

`minepy` (MIC), `boruta_py` and `tensorflow` are optional. Without them the
pipeline substitutes a kNN mutual-information estimator for MIC, a built-in
shadow-feature implementation for Boruta, and scikit-learn's `MLPClassifier`
(same 150-50-20 architecture) for the Keras network. If you install them,
swap the implementations in `fs_methods.py` and re-run; state whichever you
used in the Methods section.

## Run

Every step checkpoints, so the study can be run in slices:

```
python3 step.py fold 0      # ... through fold 4: leak-free outer CV
python3 step.py discovery   # consensus ranking on the full dataset
python3 step.py loso 0      # ... through loso 3: leave-one-study-out
python3 step.py de          # differential expression + BH-FDR
python3 step.py singlegene  # proposed vs previously published markers
python3 step.py null        # random-signature null model
python3 step.py collect     # assemble results/table*.csv
python3 extra_analyses.py composition
python3 extra_analyses.py network
python3 extra_analyses.py roc
python3 null_dist.py
python3 review_checks.py part1   # selection bias, method overlap, nested null model
python3 review_checks.py part2   # nested comparison with published markers
python3 review_checks.py part3   # nested composition adjustment, consensus ablation
python3 figures.py fig1 fig2 fig3 fig4 fig5 fig6 fig7 fig8 fig9 figS1 figS2
MELANOMA_DATA=path/to/Data.csv python3 enrichment_ppi.py   # online
```

### Paths

All scripts read paths from `config.py`. Put the matrix at `data/Data.csv`
(or `Main/Data.csv`, the thesis layout), or point to it explicitly:

```
export MELANOMA_DATA=/path/to/Data.csv      # Windows: set MELANOMA_DATA=...
export MELANOMA_OUT=/path/to/output_dir     # optional; default = this folder
```

### Robustness analysis (reference implementations)

```
python3 sensitivity.py fold 0      # ... through fold 4
python3 sensitivity.py mlpall 0    # ... through mlpall 4
python3 sensitivity.py discovery
python3 sensitivity.py collect     # -> results/table12_sensitivity.json
```

`fs_methods_faithful.py` holds the full Boruta algorithm, MIC computed
exactly for a categorical target (dynamic programming, checked against
exhaustive search), and a NumPy re-implementation of the Keras network with
dropout. They need only NumPy, SciPy and scikit-learn.

## Files

| file | purpose |
|---|---|
| `core.py` | data loading, symbol repair, study inference, metrics |
| `fs_methods.py` | 32 feature-selection methods, one common signature |
| `step.py` | resumable driver for every analysis stage |
| `config.py` | data and output paths (env vars MELANOMA_DATA, MELANOMA_OUT) |
| `fs_methods_faithful.py` | full Boruta, exact MIC, Keras-equivalent dropout network |
| `revision_analyses.py` | uncertainty and corrected tests, classifier dependence, in-fold batch correction, duplicate screen |
| `mcpcounter_adjust.py` | MCP-counter (published markers) composition adjustment |
| `repeated_cv_summary.py` | 3 x 5 repeated nested CV summary (set MELANOMA_CV_SEED for step.py) |
| `tcga_validate_cbio.py` | external TCGA-SKCM validation from the saved cBioPortal values (results/revision/tcga_skcm_signature_rnaseq.tsv) |
| `tcga_validation.py` | external primary-vs-metastatic validation on TCGA-SKCM (needs internet) |
| `geo_patient_check.py` | patient identifiers from GEO sample records (needs internet) |
| `deconvolution.R`, `deconv_adjust.py` | xCell / MCP-counter composition estimates and re-adjustment (needs internet) |
| `audit_numbers.py`, `reproduce_all.sh` | number audit; one-command reproduction |
| `requirements-lock.txt` | exact package versions used |
| `review_checks.py` | selection bias, stability, ablation; nested null, published-marker and composition analyses |
| `sensitivity.py` | robustness of results to those three implementations |
| `fs_methods_reference.py` | minepy MIC, boruta_py, Keras; auto-activate when installed |
| `extra_analyses.py` | composition adjustment, co-expression network, ROC |
| `enrichment_ppi.py` | GO/KEGG enrichment and STRING network (needs internet) |
| `null_dist.py` | saves the 500-set random-signature distribution |
| `figures.py` | all manuscript figures, 300-dpi PNG + vector PDF |

## What changed relative to the original implementation

1. **Selection bias removed.** Selection used to be fitted on the whole
   matrix before cross-validation, so every test fold had been seen during
   selection. Selection now happens inside each training fold. On this data
   the old arrangement inflated accuracy by roughly eight percentage points.
2. **Folds are shuffled** (`shuffle=True, random_state=42`). Samples in
   `Data.csv` are ordered by class and then by source study, so unshuffled
   folds were contiguous blocks of a single study.
3. **Sensitivity** is macro recall. The previous `sensitivity_score`
   computed `cm.diagonal().sum() / cm.sum()`, which is overall accuracy; the
   reported Sensitivity column equalled Accuracy in 30 of 33 result files.
4. **Specificity** is one-vs-rest, averaged over the three classes. The
   previous version read `cm[0,0]` and `cm[0,1]` only, giving a single
   class' value on a 3x3 matrix.
5. **Seeds** are set for every stochastic method.
6. **Redundancy is handled automatically.** Methods whose selected sets
   overlap above Jaccard 0.80 are clustered and share one vote, instead of
   near-duplicate methods each contributing a full vote. In the original
   outputs ANOVA and Pearson selected identical sets, as did ICHI,
   Information Gain and Mutual Information.
7. **The consensus signature is re-derived inside every outer fold**, so the
   reported signature performance is not circular.
8. **The neural network uses the selected genes.** The original script
   contained `selected_features = selected_features[:2]` under a comment
   saying 15, so the published comparison was a two-gene model.
9. **Gene symbols repaired.** Excel had converted 18 symbols to dates
   (`Mar-01` to `MARCHF1`, `Sep-02` to `SEPTIN2`); two were duplicated and
   are now collapsed. `LPPR4` is renamed to its current symbol `PLPPR4`.
10. **Added analyses:** leave-one-study-out validation, Benjamini-Hochberg
    FDR across all 12,337 genes, a random-signature null model, and
    mRMR/MIFS/Kruskal-Wallis/Extra Trees/PSO/differential evolution, whose
    absence a reviewer would likely query.

## Outputs

| file | content |
|---|---|
| `table1_method_benchmark.csv` | per-method metrics, leak-free, mean over 5 folds |
| `table2_gene_ranking.csv` | consensus score per gene, full-data discovery run |
| `table3_model_comparison.csv` | consensus-15 vs all-genes, KNN and MLP, nested |
| `table4_leave_one_study_out.csv` | cross-study generalisation |
| `table5_single_gene.csv` | single-gene power, proposed vs prior markers |
| `table6_null_model.json` | observed vs 500 random 5-gene signatures |
| `table7_signature_stability.csv` | how often each gene survives across folds |
| `differential_expression_all_genes.csv` | log2FC, p, FDR for three contrasts |
| `method_redundancy.json` | which methods were found redundant |
| `cleaning_report.json` | what the cleaning step changed |


Patient-grouped CV: `MELANOMA_GROUPED=1 python3 step.py fold N` uses results/revision/patient_groups.csv (built from GEO sample records, results/revision/geo_sample_characteristics.csv).
Results reproduce exactly with Python 3.12.3/scikit-learn 1.8.0 and Python 3.13/scikit-learn 1.9.1 (fold 0 checked).
