"""
Number audit: every key figure quoted in the manuscript is recomputed from the
result files written by the analysis code, or from Data.csv itself, and
compared with the value printed in the text.

    python3 audit_numbers.py files   # checks against results/*.csv|json
    python3 audit_numbers.py data    # checks recomputed from Data.csv
Writes results/audit/audit_<mode>.csv; PASS when |recomputed - stated| <= tolerance.
"""
import os, sys, json, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, RESULTS, CKPT

AUD = os.path.join(RESULTS, "audit"); os.makedirs(AUD, exist_ok=True)
rows = []


def chk(section, claim, stated, value, tol=0.0015, source=""):
    if isinstance(stated, str):
        ok = str(value) == stated
    else:
        ok = abs(float(value) - float(stated)) <= tol
    rows.append({"section": section, "claim": claim, "stated_in_paper": stated,
                 "recomputed": value if isinstance(value, str) else round(float(value), 6),
                 "source": source, "result": "PASS" if ok else "FAIL"})


def J(p):
    return json.load(open(os.path.join(RESULTS, p)))


def files():
    t1 = pd.read_csv(f"{RESULTS}/table1_method_benchmark.csv").set_index("method")
    st = pd.read_csv(f"{RESULTS}/review/accuracy_vs_stability.csv").set_index("method")
    src = "table1_method_benchmark.csv"
    chk("3.1", "min balanced accuracy", 0.731, t1.balanced_accuracy.min(), source=src)
    chk("3.1", "max balanced accuracy", 0.910, t1.balanced_accuracy.max(), source=src)
    chk("3.1", "ACO runtime (s)", 2.9, t1.loc["ACO", "runtime_s"], tol=0.051, source=src)
    chk("3.1", "beam search runtime (s)", 26.5, t1.loc["Beam Search", "runtime_s"], tol=0.051, source=src)
    tab1 = {"ACO": (0.913, 0.910, 0.910, 0.969, 0.04), "MIFS": (0.894, 0.905, 0.901, 0.963, 0.13),
            "Forward Selection": (0.894, 0.896, 0.898, 0.968, 0.18),
            "Simulated Annealing": (0.878, 0.893, 0.888, 0.961, 0.05),
            "Boruta": (0.891, 0.893, 0.894, 0.959, 0.11), "Genetic Algorithm": (0.900, 0.891, 0.899, 0.967, 0.05),
            "Kruskal-Wallis": (0.885, 0.889, 0.882, 0.957, 0.34), "LASSO (L1)": (0.882, 0.886, 0.890, 0.960, 0.34),
            "Stability Selection": (0.888, 0.882, 0.893, 0.958, 0.51), "ANOVA": (0.838, 0.826, 0.834, 0.943, 0.33),
            "Variance Threshold": (0.797, 0.782, 0.792, 0.912, 0.93),
            "Spearman Correlation": (0.763, 0.731, 0.731, 0.876, 0.35)}
    for m, (a, b, f, auc, jac) in tab1.items():
        chk("Table 1", f"{m} accuracy", a, t1.loc[m, "accuracy"], source=src)
        chk("Table 1", f"{m} balanced acc.", b, t1.loc[m, "balanced_accuracy"], source=src)
        chk("Table 1", f"{m} macro F1", f, t1.loc[m, "f1_macro"], source=src)
        chk("Table 1", f"{m} AUC", auc, t1.loc[m, "auc_ovr"], source=src)
        chk("Table 1", f"{m} stability", jac, st.loc[m, "mean_jaccard"], tol=0.0051, source="review/accuracy_vs_stability.csv")
    top5 = t1.sort_values("balanced_accuracy", ascending=False).index[:5]
    chk("3.1", "top-5 selectors min Jaccard", 0.04, st.loc[top5, "mean_jaccard"].min(), tol=0.0051, source="accuracy_vs_stability.csv")
    chk("3.1", "top-5 selectors max Jaccard", 0.18, st.loc[top5, "mean_jaccard"].max(), tol=0.0051, source="accuracy_vs_stability.csv")
    for m, v in [("PCA Loading", 0.95), ("Chi-square", 0.88)]:
        chk("3.1", f"{m} Jaccard", v, st.loc[m, "mean_jaccard"], tol=0.0051, source="accuracy_vs_stability.csv")
    p1 = J("review/part1.json")
    chk("1 / 3.1", "median pairwise overlap (genes)", 2, p1["pairwise_overlap_median"], tol=0, source="review/part1.json")
    chk("1 / 3.1", "pairs sharing no gene", 72, p1["pairs_with_zero_overlap"], tol=0, source="review/part1.json")
    chk("1 / 3.1", "number of method pairs", 496, p1["n_pairs"], tol=0, source="review/part1.json")
    chk("3.1", "methods inflated by selection bias", 24, p1["n_methods_inflated"], tol=0, source="review/part1.json")
    chk("3.1", "median inflation (points)", 1.6, 100 * p1["inflation_median"], tol=0.051, source="review/part1.json")
    chk("3.1", "max inflation (points)", 5.9, 100 * p1["inflation_max"], tol=0.051, source="review/part1.json")
    chk("3.1", "ANOVA leaky accuracy", 0.897, p1["anova_leaky_vs_nested"][0], source="review/part1.json")
    chk("3.1", "ANOVA nested accuracy", 0.838, p1["anova_leaky_vs_nested"][1], source="review/part1.json")
    red = J("method_redundancy.json")["redundant_clusters"]
    clusters = sorted(sorted(v) for v in red.values())
    chk("3.1", "redundancy clusters", "[['ANOVA', 'Fisher Score', 'Permutation Importance'], ['PCA Loading', 'Variance Threshold']]",
        str(clusters), source="method_redundancy.json")
    sig = J("signature_stability.json")
    chk("3.2", "genes in all five folds", "['KRT14', 'PLPPR4', 'S100A7']", str(sorted(sig["in_all_folds"])), source="signature_stability.json")
    t7 = pd.read_csv(f"{RESULTS}/table7_signature_stability.csv").set_index("gene")
    chk("3.2", "genes in four folds", "['KRT15', 'KRT16', 'KRT6A', 'SERPINB5']",
        str(sorted(t7[t7.folds_selected_of_5 == 4].index)), source="table7_signature_stability.csv")
    rank = pd.read_csv(f"{RESULTS}/table2_gene_ranking.csv", index_col=0)
    chk("3.2", "full-data top 10", "['PLPPR4', 'KRT6A', 'KRT14', 'EMP3', 'ANXA8', 'KRT16', 'S100A7', 'SERPINB5', 'KLF5', 'AHNAK']",
        str(list(rank.index[:10])), source="table2_gene_ranking.csv")
    de = pd.read_csv(f"{RESULTS}/differential_expression_all_genes.csv").set_index("gene")
    t2 = {"PLPPR4": (5, 18.60, -0.17), "S100A7": (5, 5.88, -4.44), "KRT14": (5, 9.28, -6.72), "KRT6A": (4, 10.75, -4.89),
          "KRT15": (4, 3.08, -6.51), "SERPINB5": (4, 5.65, -5.47), "KRT16": (4, 6.69, -3.79), "EMP3": (3, 7.12, 1.00),
          "TSPAN8": (3, 3.55, -2.11), "AHNAK": (3, 5.05, -2.15)}
    for g, (nf, sc, fc) in t2.items():
        chk("Table 2", f"{g} folds", nf, t7.loc[g, "folds_selected_of_5"], tol=0, source="table7_signature_stability.csv")
        chk("Table 2", f"{g} consensus score", sc, rank.loc[g, "consensus_score"], tol=0.0051, source="table2_gene_ranking.csv")
        chk("Table 2", f"{g} log2FC met vs normal", fc, de.loc[g, "log2FC_meta_vs_healthy"], tol=0.0051, source="differential_expression_all_genes.csv")
    chk("Table 2", "PLPPR4 FDR met vs normal", 0.462, de.loc["PLPPR4", "fdr_meta_vs_healthy"], tol=0.0006, source="differential_expression_all_genes.csv")
    chk("1", "genes with FDR<0.05 met vs normal", 6600, int((de.fdr_meta_vs_healthy < 0.05).sum()), tol=0, source="differential_expression_all_genes.csv")
    chk("3.6", "PLPPR4 FDR primary vs normal", 8.2e-10, de.loc["PLPPR4", "fdr_atrisk_vs_healthy"], tol=0.051e-10, source="differential_expression_all_genes.csv")
    chk("3.6", "PLPPR4 FDR met vs primary", 2.8e-27, de.loc["PLPPR4", "fdr_meta_vs_atrisk"], tol=0.051e-27, source="differential_expression_all_genes.csv")
    m = pd.concat([t1["balanced_accuracy"], st["mean_jaccard"]], axis=1)
    chk("3.2", "median selector balanced accuracy", 0.873, m.balanced_accuracy.median(), source="table1 + stability")
    chk("3.2", "median selector Jaccard", 0.20, m.mean_jaccard.median(), tol=0.0051, source="table1 + stability")
    p3 = J("review/part3.json")
    chk("3.2", "consensus Jaccard", 0.29, p3["ablation"]["weighted+redundancy"]["mean_pairwise_jaccard"], tol=0.0051, source="review/part3.json")
    for name, (acc, bal, core, jac) in {"weighted+redundancy": (0.888, 0.873, 3, 0.287), "weighted only": (0.891, 0.858, 3, 0.301),
                                        "redundancy only": (0.885, 0.871, 4, 0.308), "plain vote count": (0.894, 0.860, 3, 0.307)}.items():
        a = p3["ablation"][name]
        chk("Table S2", f"{name} accuracy", acc, a["accuracy"], source="review/part3.json")
        chk("Table S2", f"{name} balanced acc.", bal, a["balanced_accuracy"], source="review/part3.json")
        chk("Table S2", f"{name} genes in all folds", core, a["genes_in_all_5_folds"], tol=0, source="review/part3.json")
        chk("Table S2", f"{name} Jaccard", jac, a["mean_pairwise_jaccard"], source="review/part3.json")
    t3 = pd.read_csv(f"{RESULTS}/table3_model_comparison.csv").set_index(["model", "features"])
    for (mdl, feat), vals in {("KNN", "consensus-15"): (0.888, 0.873, 0.885, 0.932, 0.957),
                              ("KNN", "all-genes"): (0.850, 0.859, 0.859, 0.909, 0.962),
                              ("MLP (Keras-equivalent)", "consensus-15"): (0.897, 0.907, 0.906, 0.939, 0.951),
                              ("MLP (Keras-equivalent)", "all-genes"): (0.850, 0.848, 0.835, 0.914, 0.947)}.items():
        r = t3.loc[(mdl, feat)]
        for k, v in zip(["accuracy", "balanced_accuracy", "f1_macro", "specificity_macro", "auc_ovr"], vals):
            chk("Table 3", f"{mdl} {feat} {k}", v, r[k], source="table3_model_comparison.csv")
    chk("3.3", "sklearn MLP consensus balanced acc.", 0.772, t3.loc[("MLP (sklearn, no dropout)", "consensus-15"), "balanced_accuracy"], source="table3")
    chk("3.3", "sklearn MLP all-genes balanced acc.", 0.833, t3.loc[("MLP (sklearn, no dropout)", "all-genes"), "balanced_accuracy"], source="table3")
    roc = J("roc_out_of_fold.json")
    for c, v in [("1", 0.948), ("2", 0.955), ("3", 0.965)]:
        chk("3.3", f"out-of-fold AUC class {c}", v, roc[c]["auc"], source="roc_out_of_fold.json")
    nn = J("revision/stats.json")["null_5000"]
    for k, v, tol in [("observed", 0.843, 0.0015), ("null_mean", 0.512, 0.0015), ("null_p95", 0.645, 0.0015),
                      ("null_max", 0.791, 0.0015), ("empirical_p", 0.0002, 0.00005)]:
        chk("3.4", f"null model (5,000 draws) {k}", v, nn[k], tol=tol, source="revision/stats.json")
    st = J("revision/stats.json")
    c = st["consensus"]
    chk("3.10", "consensus balanced acc. mean", 0.873, c["bal_acc_mean"], source="revision/stats.json")
    chk("3.10", "consensus balanced acc. SD", 0.066, c["bal_acc_sd"], source="revision/stats.json")
    chk("3.10", "consensus 95% CI low", 0.79, c["bal_acc_ci_low"], tol=0.0051, source="revision/stats.json")
    chk("3.10", "consensus 95% CI high", 0.96, c["bal_acc_ci_high"], tol=0.0051, source="revision/stats.json")
    chk("3.10", "consensus Jaccard mean", 0.29, c["jaccard_mean"], tol=0.0051, source="revision/stats.json")
    chk("3.10", "consensus Jaccard SD", 0.09, c["jaccard_sd"], tol=0.0051, source="revision/stats.json")
    tt = st["corrected_tests_vs_consensus"]
    chk("3.10", "min corrected-test p > 0.13", "True", str(min(v["p"] for v in tt.values()) > 0.13), source="revision/stats.json")
    chk("3.10", "min Holm-adjusted p >= 0.92", "True", str(round(min(v["p_holm"] for v in tt.values()), 2) >= 0.92), source="revision/stats.json")
    cl = J("revision/classifiers.json")
    chk("3.10", "consensus rank kNN", 17, cl["kNN"]["consensus_rank_of_33"], tol=0, source="revision/classifiers.json")
    chk("3.10", "consensus rank logistic", 3, cl["logistic"]["consensus_rank_of_33"], tol=0, source="revision/classifiers.json")
    chk("3.10", "consensus rank random forest", 1, cl["random_forest"]["consensus_rank_of_33"], tol=0, source="revision/classifiers.json")
    chk("3.10", "consensus logistic balanced acc.", 0.886, cl["logistic"]["consensus_bal_acc"], source="revision/classifiers.json")
    chk("3.10", "consensus RF balanced acc.", 0.932, cl["random_forest"]["consensus_bal_acc"], source="revision/classifiers.json")
    chk("3.10", "rank correlation kNN vs logistic", 0.75, cl["rank_correlation_kNN_vs_logistic"], tol=0.0051, source="revision/classifiers.json")
    chk("3.10", "rank correlation kNN vs RF", 0.73, cl["rank_correlation_kNN_vs_rf"], tol=0.0051, source="revision/classifiers.json")
    s4 = pd.read_csv(f"{RESULTS}/revision/tableS4_classifier_dependence.csv", index_col=0)
    tS4 = {"CONSENSUS": (0.873, 0.886, 0.932, 0.897, 17, 3, 1), "Genetic Algorithm": (0.891, 0.893, 0.903, 0.896, 6, 1, 5),
           "Stability Selection": (0.882, 0.874, 0.924, 0.894, 12, 6, 2), "mRMR": (0.883, 0.870, 0.921, 0.891, 11, 7, 3),
           "Simulated Annealing": (0.893, 0.861, 0.901, 0.885, 4, 9, 8), "Boruta": (0.893, 0.858, 0.903, 0.885, 5, 13, 6),
           "ACO": (0.910, 0.864, 0.878, 0.884, 1, 8, 17), "LASSO (L1)": (0.886, 0.860, 0.898, 0.881, 8, 11, 10),
           "Kruskal-Wallis": (0.889, 0.849, 0.843, 0.860, 7, 17, 26)}
    for m, v in tS4.items():
        for col, x, tol in zip(["kNN", "logistic", "random_forest", "mean", "rank_kNN", "rank_LR", "rank_RF"], v, [0.0015]*4 + [0]*3):
            chk("Table S4", f"{m} {col}", x, s4.loc[m, col], tol=tol, source="revision/tableS4_classifier_dependence.csv")
    bt = J("revision/batch.json")
    chk("3.10", "pooled balanced acc. uncorrected", 0.873, bt["pooled_cv_consensus_bal_acc"]["uncorrected"], source="revision/batch.json")
    chk("3.10", "pooled balanced acc. corrected", 0.904, bt["pooled_cv_consensus_bal_acc"]["in_fold_corrected"], source="revision/batch.json")
    chk("3.10", "study prediction before", 0.63, bt["study_prediction_accuracy"]["uncorrected"], tol=0.0051, source="revision/batch.json")
    chk("3.10", "study prediction after", 0.72, bt["study_prediction_accuracy"]["in_fold_corrected"], tol=0.0051, source="revision/batch.json")
    for k, (a, c_) in {"study_1": (0.46, 0.52), "study_2": (0.86, 0.88), "study_3": (0.66, 0.56), "study_4": (0.85, 0.83)}.items():
        chk("3.10", f"LOSO {k} uncorrected", a, bt["loso_study_centring"][k]["uncorrected"], tol=0.0051, source="revision/batch.json")
        chk("3.10", f"LOSO {k} study-centred", c_, bt["loso_study_centring"][k]["study_centred"], tol=0.0051, source="revision/batch.json")
    du = J("revision/duplicates.json")
    chk("3.10", "pairs with r >= 0.98", 1, du["n_pairs_above"], tol=0, source="revision/duplicates.json")
    chk("3.10", "max pairwise r", 0.994, du["max_r"], source="revision/duplicates.json")
    chk("3.10", "99th percentile r", 0.93, du["pct99_r"], tol=0.0051, source="revision/duplicates.json")
    rc = J("revision/repeated_cv.json")
    chk("3.10", "repeated CV outer folds", 15, rc["n_outer_folds"], tol=0, source="revision/repeated_cv.json")
    chk("3.10", "repeated CV consensus mean", 0.871, rc["consensus"]["bal_mean"], source="revision/repeated_cv.json")
    chk("3.10", "repeated CV consensus SD", 0.062, rc["consensus"]["bal_sd"], source="revision/repeated_cv.json")
    chk("3.10", "repeated CV CI low", 0.84, rc["consensus"]["ci_low"], tol=0.0051, source="revision/repeated_cv.json")
    chk("3.10", "repeated CV CI high", 0.91, rc["consensus"]["ci_high"], tol=0.0051, source="revision/repeated_cv.json")
    chk("3.10", "repeated CV consensus stability", 0.35, rc["consensus"]["jaccard"], tol=0.0051, source="revision/repeated_cv.json")
    chk("3.10", "repeated CV median selector", 0.870, rc["median_selector_bal"], source="revision/repeated_cv.json")
    chk("3.10", "repeated CV median stability", 0.23, rc["median_selector_jaccard"], tol=0.0051, source="revision/repeated_cv.json")
    chk("3.10", "repeated CV consensus rank", 17, rc["consensus_rank_of_33_kNN"], tol=0, source="revision/repeated_cv.json")
    chk("3.10", "repeated CV top method", "Forward Selection", rc["top_method"], source="revision/repeated_cv.json")
    chk("3.10", "repeated CV top method value", 0.900, rc["top_method_bal"], source="revision/repeated_cv.json")
    chk("3.10", "smallest Holm-adjusted p", 0.08, min(v["p_holm"] for v in rc["corrected_tests"].values()), tol=0.0051, source="revision/repeated_cv.json")
    chk("3.10", "PLPPR4 ranked first in folds", 15, rc["plppr4_rank1_in_folds"], tol=0, source="revision/repeated_cv.json")
    chk("3.10", "genes in all 15 folds", "['PLPPR4']", str(rc["genes_in_all_15"]), source="revision/repeated_cv.json")
    chk("3.10", "genes in >= 12 folds", "['KRT14', 'KRT16', 'KRT6A', 'PLPPR4', 'S100A7', 'SERPINB5']", str(rc["genes_in_at_least_12"]), source="revision/repeated_cv.json")
    mc = J("revision/mcpcounter.json")
    chk("2.7", "MCP-counter markers found", 94, sum(mc["markers_found"].values()), tol=0, source="revision/mcpcounter.json")
    chk("2.7", "MCP-counter markers total", 111, sum(mc["markers_total"].values()), tol=0, source="revision/mcpcounter.json")
    for task, var, key, v in [("three_class", "mcp_counter_10", "signature_adjusted", 0.702),
                              ("three_class", "mcp_counter_plus_keratinocyte_melanocyte", "signature_adjusted", 0.635),
                              ("three_class", "mcp_counter_10", "scores_alone", 0.749),
                              ("primary_vs_metastatic", "mcp_counter_10", "signature_adjusted", 0.767),
                              ("primary_vs_metastatic", "mcp_counter_plus_keratinocyte_melanocyte", "signature_adjusted", 0.737)]:
        chk("3.10", f"MCP-counter {task} {var} {key}", v, mc[task][var][key], source="revision/mcpcounter.json")
    gc = J("revision/grouped_cv.json")
    chk("3.10", "grouped CV consensus balanced acc.", 0.888, gc["consensus_bal_mean"], source="revision/grouped_cv.json")
    chk("3.10", "grouped CV consensus SD", 0.038, gc["consensus_bal_sd"], source="revision/grouped_cv.json")
    chk("3.10", "grouped CV PLPPR4 first in folds", 5, gc["plppr4_rank1_folds"], tol=0, source="revision/grouped_cv.json")
    chk("3.10", "grouped CV genes in all folds", "['ANXA8', 'KRT14', 'KRT16', 'PLPPR4', 'S100A7', 'SERPINB5']", str(gc["genes_in_all_5"]), source="revision/grouped_cv.json")
    pg = pd.read_csv(f"{RESULTS}/revision/patient_groups.csv", index_col=0).patient_group
    vc = pg.value_counts()
    chk("2.7", "GSE15605 same-patient pairs", 11, int((vc[vc.index.str.startswith("GSE15605")] == 2).sum()), tol=0, source="revision/patient_groups.csv")
    m46 = vc[vc.index.str.startswith("GSE46517")]
    chk("2.7", "GSE46517 multi-metastasis patients", 9, int((m46 > 1).sum()), tol=0, source="revision/patient_groups.csv")
    chk("2.7", "samples in those patients", 21, int(m46[m46 > 1].sum()), tol=0, source="revision/patient_groups.csv")
    tc = J("revision/tcga_validation.json")
    chk("3.11", "TCGA primary n", 103, tc["n_tcga_primary"], tol=0, source="revision/tcga_validation.json")
    chk("3.11", "TCGA metastatic n", 369, tc["n_tcga_metastatic"], tol=0, source="revision/tcga_validation.json")
    chk("3.11", "TCGA kNN AUC", 0.796, tc["kNN"]["auc"], source="revision/tcga_validation.json")
    chk("3.11", "TCGA logistic AUC", 0.842, tc["logistic"]["auc"], source="revision/tcga_validation.json")
    chk("3.11", "TCGA balanced acc. kNN", 0.746, tc["kNN"]["balanced_accuracy"], source="revision/tcga_validation.json")
    chk("3.11", "TCGA balanced acc. logistic", 0.746, tc["logistic"]["balanced_accuracy"], source="revision/tcga_validation.json")
    chk("3.11", "direction agreement (of 15)", 13, tc["direction_agreement"], tol=0, source="revision/tcga_validation.json")
    chk("3.11", "PLPPR4 TCGA difference", -0.79, tc["per_gene"]["PLPPR4"]["tcga_diff"], tol=0.0051, source="revision/tcga_validation.json")
    chk("3.11", "PLPPR4 TCGA p", 5.4e-05, tc["per_gene"]["PLPPR4"]["tcga_p"], tol=0.6e-5, source="revision/tcga_validation.json")
    chk("3.11", "EMP3 TCGA p", 7e-10, tc["per_gene"]["EMP3"]["tcga_p"], tol=0.6e-10, source="revision/tcga_validation.json")
    ss = J("revision/string_summary.json")
    chk("3.11", "STRING nodes", 21, ss["nodes"], tol=0, source="revision/string_summary.json")
    chk("3.11", "STRING edges", 43, ss["edges"], tol=0, source="revision/string_summary.json")
    chk("3.11", "STRING largest module size", 15, len(ss["components"][0]), tol=0, source="revision/string_summary.json")
    chk("3.11", "candidates without partner", "True", str(all(g in ss["not_connected"] for g in ["PLPPR4", "EMP3", "FABP4", "PLOD3", "CDC42EP3"])), source="revision/string_summary.json")
    chk("3.11", "most central nodes", "['DSP', 'LOR', 'SPRR1B']", str([r["gene"] for r in ss["top_betweenness"][:3]]), source="revision/string_summary.json")
    pp = J("preprocessing/summary.json")
    best = {r["series"]: r for r in pp["best_rule_per_series"]}
    chk("2.1", "best rule is log2 averaging (3 series)", "True", str(all(best[s]["rule"] == "mean" and best[s]["averaged_on"] == "log2" for s in ["GSE3189", "GSE8401", "GSE46517"])), source="preprocessing/summary.json")
    chk("2.1", "log2-mean rho low", 0.96, min(best[s]["median_rho"] for s in ["GSE3189", "GSE8401", "GSE46517"]), tol=0.0051, source="preprocessing/summary.json")
    chk("2.1", "log2-mean rho high", 0.98, max(best[s]["median_rho"] for s in ["GSE3189", "GSE8401", "GSE46517"]), tol=0.0051, source="preprocessing/summary.json")
    bp = pp["best_probe_rule_median_rho"]
    chk("2.1", "best-probe rho low", 0.86, min(bp[s] for s in ["GSE3189", "GSE8401", "GSE46517"]), tol=0.0051, source="preprocessing/summary.json")
    chk("2.1", "best-probe rho high", 0.88, max(bp[s] for s in ["GSE3189", "GSE8401", "GSE46517"]), tol=0.0051, source="preprocessing/summary.json")
    sp = pp["single_probe_gene_median_rho"]
    chk("2.1", "single-probe rho low (3 series)", 0.95, min(sp[s] for s in ["GSE3189", "GSE8401", "GSE46517"]), tol=0.0051, source="preprocessing/summary.json")
    chk("2.1", "single-probe rho high", 0.97, max(sp[s] for s in ["GSE3189", "GSE8401", "GSE46517"]), tol=0.0051, source="preprocessing/summary.json")
    chk("2.1", "single-probe rho GSE15605", 0.88, sp["GSE15605"], tol=0.0051, source="preprocessing/summary.json")
    chk("2.1", "genes matching GPL96 symbols", 11469, pp["genes_matching_current_GPL96_symbols"], tol=0, source="preprocessing/summary.json")
    ds = J("revision/duplicate_sensitivity.json")
    chk("3.10", "balanced acc. without the pair", 0.875, ds["without_pair"], source="revision/duplicate_sensitivity.json")
    p2 = J("review/part2.json")["single_gene_nested"]
    chk("3.4", "proposed genes mean F1", 0.485, p2["proposed_mean"], source="review/part2.json")
    chk("3.4", "published markers mean F1", 0.441, p2["prior_mean"], source="review/part2.json")
    chk("3.4", "Mann-Whitney p", 0.0004, p2["mannwhitney_p"], tol=0.00005, source="review/part2.json")
    pf = p2["per_fold_means"]
    chk("3.4", "proposed higher in all 5 folds", "True", str(all(pf["proposed"][f] > pf["prior"][f] for f in pf["prior"])), source="review/part2.json")
    t4 = pd.read_csv(f"{RESULTS}/table4_leave_one_study_out.csv")
    t4 = t4[t4.model == "KNN"].set_index("held_out_study")
    for s, (n, bal, f1) in {0: (52, 0.457, 0.385), 1: (83, 0.859, 0.579), 2: (74, 0.660, 0.585), 3: (112, 0.848, 0.842)}.items():
        chk("Table 4", f"study {s+1} n", n, t4.loc[s, "n_test"], tol=0, source="table4_leave_one_study_out.csv")
        chk("Table 4", f"study {s+1} balanced acc.", bal, t4.loc[s, "balanced_accuracy"], source="table4")
        chk("Table 4", f"study {s+1} macro F1", f1, t4.loc[s, "f1_macro"], source="table4")
    for k, (a, b) in {"signature": (0.873, 0.899), "adj_keratinocyte": (0.864, 0.822), "adj_all": (0.665, 0.796),
                      "keratinocyte_only": (0.481, 0.655), "all_scores_only": (0.874, 0.855)}.items():
        chk("Table 5", f"{k} three-class", a, p3["three_class"][k], source="review/part3.json")
        chk("Table 5", f"{k} primary vs metastatic", b, p3["primary_vs_metastatic"][k], source="review/part3.json")
    cs = J("review/cell_scores_by_class.json")
    chk("3.7", "normal skin keratinocyte score", 0.92, cs["keratinocyte"]["normal skin"], tol=0.0051, source="review/cell_scores_by_class.json")
    chk("3.7", "normal skin melanocyte score", -1.16, cs["melanocyte"]["normal skin"], tol=0.0051, source="review/cell_scores_by_class.json")
    chk("3.7", "metastatic keratinocyte score", -0.58, cs["keratinocyte"]["metastatic melanoma"], tol=0.0051, source="review/cell_scores_by_class.json")
    net = J("table9_network_summary.json")
    chk("3.8", "network nodes", 41, net["nodes"], tol=0, source="table9_network_summary.json")
    chk("3.8", "network edges", 574, net["edges"], tol=0, source="table9_network_summary.json")
    chk("3.8", "component sizes", "[38, 3]", str([len(c) for c in net["components"]]), source="table9_network_summary.json")
    chk("3.8", "isolated genes", 19, len(net["isolated_removed"]), tol=0, source="table9_network_summary.json")
    nt = pd.read_csv(f"{RESULTS}/table9_coexpression_network.csv", index_col=0)
    chk("3.8", "median degree of epidermal module", 32, nt[nt.component == 0].degree.median(), tol=0, source="table9_coexpression_network.csv")
    sens = J("table12_sensitivity.json")
    chk("Table S3", "fold overlaps with main signature", "[13, 13, 13, 14, 13]", str(sens["per_fold_overlap_with_original"]), source="table12_sensitivity.json")
    chk("Table S3", "full Boruta balanced acc.", 0.882, sens["boruta_full_bal_acc"], source="table12_sensitivity.json")
    chk("Table S3", "exact MIC balanced acc.", 0.861, sens["mic_exact_bal_acc"], source="table12_sensitivity.json")
    chk("Table S3", "reference kNN macro F1", 0.875, sens["knn_consensus_reference"]["f1_macro"], source="table12_sensitivity.json")
    dc = J("sensitivity/discovery_comparison.json")
    chk("Table S3", "top-15 overlap", 14, dc["overlap_top15"], tol=0, source="sensitivity/discovery_comparison.json")
    chk("Table S3", "top-60 overlap", 56, dc["overlap_top60"], tol=0, source="sensitivity/discovery_comparison.json")


def data():
    from core import load_data
    from scipy import stats
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    X, y, b, rep = load_data(DATA, verbose=False)
    src = "Data.csv"
    chk("2.1", "genes after cleaning", 12337, X.shape[1], tol=0, source=src)
    chk("2.1", "samples", 321, X.shape[0], tol=0, source=src)
    for c, n in [(1, 31), (2, 153), (3, 137)]:
        chk("2.1", f"class {c} size", n, int((y == c).sum()), tol=0, source=src)
    ct = pd.crosstab(b.values, y.values)
    for s, vals in enumerate([(7, 45, 0), (0, 31, 52), (16, 46, 12), (8, 31, 73)]):
        for c, v in zip([1, 2, 3], vals):
            chk("Table S1", f"study {s+1} class {c}", v, int(ct.loc[s, c]) if c in ct.columns else 0, tol=0, source=src)
    raw = pd.read_csv(DATA, index_col=0, usecols=[0]).index
    dates = [g for g in raw if re.match(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d+$", str(g))]
    chk("2.1", "rows with date-converted symbols", 18, len(dates), tol=0, source=src)
    p = X["PLPPR4"]
    for c, v in [(1, 6.69), (2, 8.34), (3, 6.52)]:
        chk("3.6", f"PLPPR4 mean class {c}", v, p[y == c].mean(), tol=0.0051, source=src)
    e = X["EMP3"]
    for c, v in [(1, 9.63), (2, 11.44), (3, 10.63)]:
        chk("3.6", f"EMP3 mean class {c}", v, e[y == c].mean(), tol=0.0051, source=src)
    chk("3.6", "PLPPR4 SD of study means", 0.17, p.groupby(b.values).mean().std(), tol=0.0051, source=src)
    chk("3.6", "PLPPR4 mean within-study SD", 1.49, p.groupby(b.values).std().mean(), tol=0.0051, source=src)
    pooled = stats.ttest_ind(p[y != 1], p[y == 1], equal_var=False).pvalue
    chk("3.6", "PLPPR4 pooled tumour vs normal p (order of magnitude)", -4, np.floor(np.log10(pooled)), tol=0, source=src)
    S = pd.read_csv(f"{RESULTS}/selected_features_per_method.csv")
    chk("1 / 4.2", "selectors choosing PLPPR4", 23, sum("PLPPR4" in set(S[c].dropna()) for c in S.columns), tol=0, source="selected_features_per_method.csv")
    Z = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    sc = PCA(5, random_state=0).fit_transform(Z.values)[:, 0]

    def eta2(v, lab):
        lab = np.asarray(lab); tot = ((v - v.mean()) ** 2).sum()
        return sum(len(v[lab == g]) * (v[lab == g].mean() - v.mean()) ** 2 for g in np.unique(lab)) / tot
    chk("2.1", "PC1 variance explained by class (%)", 38, 100 * eta2(sc, y.values), tol=0.51, source=src)
    chk("2.1", "PC1 variance explained by study (%)", 1, 100 * eta2(sc, b.values), tol=0.51, source=src)
    main_folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X.values, y.values))  # the paper's folds
    acc = float(np.mean([(RandomForestClassifier(200, random_state=0, n_jobs=-1).fit(X.values[tr], b.values[tr])
                          .predict(X.values[te]) == b.values[te]).mean() for tr, te in main_folds]))
    chk("2.1", "RF accuracy predicting study (main folds)", 0.63, acc, tol=0.0051, source=src)
    chk("2.1", "majority baseline for study", 0.35, b.value_counts(normalize=True).max(), tol=0.0051, source=src)


if __name__ == "__main__":
    mode = sys.argv[1]
    {"files": files, "data": data}[mode]()
    d = pd.DataFrame(rows)
    d.to_csv(f"{AUD}/audit_{mode}.csv", index=False)
    print(f"{mode}: {len(d)} checks, {(d.result == 'PASS').sum()} PASS, {(d.result == 'FAIL').sum()} FAIL")
    if (d.result == "FAIL").any():
        print(d[d.result == "FAIL"].to_string(index=False))
