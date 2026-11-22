"""Publication figures. Each figure is saved as 300-dpi PNG and vector PDF."""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import load_data

from config import DATA, RESULTS as R, FIGURES as F

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "figure.dpi": 110,
})
# Okabe-Ito, colour-blind safe
FAM = {"metaheuristic": "#D55E00", "information": "#CC79A7", "wrapper": "#E69F00",
       "embedded": "#0072B2", "filter": "#56B4E9", "projection": "#999999"}
CLS = {1: "#009E73", 2: "#E69F00", 3: "#D55E00"}
CLSN = {1: "Normal skin", 2: "Primary melanoma", 3: "Metastatic melanoma"}
STUDY = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]


def save(fig, name):
    fig.savefig(f"{F}/{name}.png", bbox_inches="tight")
    fig.savefig(f"{F}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 54); ax.axis("off")

    def box(x, y, w, h, text, fc="#F2F2F2", ec="#333333", bold=False, fs=6.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#333333"))

    box(0.5, 30, 19, 15, "4 GEO studies\nGSE3189, GSE8401\nGSE15605, GSE46517")
    box(0.5, 7, 19, 16, "Merge, quantile\nnormalisation,\nsymbol repair\n12,337 genes\n321 samples")
    arrow(10, 29.6, 10, 23.6)
    ax.add_patch(FancyBboxPatch((23, 3), 51, 43, boxstyle="round,pad=0.3,rounding_size=1.5",
                                fc="white", ec="#0072B2", lw=1.2, ls="--"))
    ax.text(48.5, 47.6, "Inside each outer training fold (5-fold, shuffled, seed 42)",
            ha="center", fontsize=7.6, color="#0072B2", fontweight="bold")
    box(25, 28, 22, 14, "32 selectors\n6 method families\ntop 15 genes each")
    box(51, 28, 21, 14, "Weight = inner-CV\naccuracy of each set\n(training data only)")
    box(25, 7, 22, 14, "Redundancy clustering\nJaccard ≥ 0.80\none shared vote")
    box(51, 7, 21, 14, "Consensus score\nΣ weight × accuracy\ntop-15 signature",
        fc="#E6F0F8", bold=True)
    arrow(20, 15, 24.6, 34)
    arrow(47.4, 35, 50.6, 35)
    arrow(36, 27.6, 36, 21.4)
    arrow(47.4, 14, 50.6, 14)
    arrow(61.5, 27.6, 61.5, 21.4)
    box(78, 34, 21.5, 11, "Held-out fold\nkNN and MLP\nvs all genes", fc="#FDF0E6")
    box(78, 19.5, 21.5, 11, "Leave-one-study-out\nnull model (500 sets)\nprior markers", fc="#FDF0E6")
    box(78, 5, 21.5, 11, "Composition adjustment\nco-expression network\nROC", fc="#FDF0E6")
    arrow(72.4, 14, 77.6, 39.5)
    arrow(72.4, 14, 77.6, 25)
    arrow(72.4, 14, 77.6, 10.5)
    save(fig, "Fig1_pipeline")


def fig2():
    d = pd.read_csv(f"{R}/table1_method_benchmark.csv").sort_values("balanced_accuracy")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 6.2), sharey=True,
                               gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.06})
    yy = np.arange(len(d))
    c = [FAM[f] for f in d.family]
    a.hlines(yy, 0.70, d.balanced_accuracy, color="#DDDDDD", lw=0.8)
    a.scatter(d.balanced_accuracy, yy, c=c, s=26, zorder=3, edgecolor="white", lw=0.4)
    a.set_yticks(yy); a.set_yticklabels(d.method)
    a.set_xlim(0.70, 0.93); a.set_xlabel("Balanced accuracy (nested 5-fold CV)")
    a.axvline(0.859, color="#333333", ls=":", lw=0.9)
    a.text(0.857, len(d) - 2.5, "kNN on all\n12,337 genes", fontsize=7, color="#333333", ha="right")
    b.hlines(yy, 0.02, d.runtime_s, color="#DDDDDD", lw=0.8)
    b.scatter(d.runtime_s, yy, c=c, s=26, zorder=3, edgecolor="white", lw=0.4)
    b.set_xscale("log"); b.set_xlabel("Runtime per fold (s, log)")
    b.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], marker="o", ls="", color=v, label=k) for k, v in FAM.items()]
    a.legend(handles=handles, loc="lower right", frameon=False, title="Family", title_fontsize=8, bbox_to_anchor=(1.0, 0.0))
    fig.suptitle("Performance of 32 feature-selection methods, leak-free", x=0.02, ha="left",
                 fontweight="bold", y=0.93)
    save(fig, "Fig2_method_benchmark")


def fig3():
    m = pd.read_csv(f"{R}/review/accuracy_vs_stability.csv")
    s = json.load(open(f"{R}/signature_stability.json"))["per_fold"]
    stab = pd.read_csv(f"{R}/table7_signature_stability.csv")
    stab = stab[stab.folds_selected_of_5 >= 2].sort_values(
        ["folds_selected_of_5", "consensus_score"], ascending=[False, False])
    genes = list(stab.gene)
    M = np.array([[1 if g in s[f] else 0 for f in range(5)] for g in genes])
    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 0.75], wspace=0.45)
    a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1])
    for fam, col in FAM.items():
        d = m[m.family == fam]
        a.scatter(d.mean_jaccard, d.balanced_accuracy, s=26, color=col, label=fam,
                  edgecolor="white", lw=0.4, zorder=3)
    a.scatter([0.287], [0.873], marker="*", s=180, color="black", zorder=4, label="consensus")
    for name in ["Kruskal-Wallis", "LASSO (L1)", "Stability Selection", "ACO", "PCA Loading",
                 "Variance Threshold", "Chi-square", "ANOVA"]:
        r = m[m.method == name].iloc[0]
        left = r.mean_jaccard > 0.7
        a.annotate(name.replace(" (L1)", "").replace(" Loading", ""), (r.mean_jaccard, r.balanced_accuracy),
                   fontsize=6.5, xytext=(-4 if left else 4, 2), textcoords="offset points",
                   ha="right" if left else "left")
    a.set_xlabel("Fold-to-fold stability (mean pairwise Jaccard of selected genes)")
    a.set_ylabel("Balanced accuracy (nested CV)")
    a.legend(frameon=False, fontsize=6.8, loc="center right", bbox_to_anchor=(1.0, 0.55))
    a.set_xlim(-0.02, 1.0)
    a.set_title("A  Accuracy against stability", loc="left", fontsize=9, fontweight="bold")
    b.imshow(M, cmap=matplotlib.colors.ListedColormap(["#F2F2F2", "#0072B2"]), aspect="auto")
    b.set_xticks(range(5)); b.set_xticklabels([f"F{i+1}" for i in range(5)])
    b.set_yticks(range(len(genes))); b.set_yticklabels(genes, fontsize=7)
    b.set_xticks(np.arange(-.5, 5, 1), minor=True); b.set_yticks(np.arange(-.5, len(genes), 1), minor=True)
    b.grid(which="minor", color="white", lw=1.5); b.tick_params(which="minor", length=0)
    for sp in b.spines.values(): sp.set_visible(False)
    b.set_title("B  Consensus genes by fold", loc="left", fontsize=9, fontweight="bold")
    save(fig, "Fig3_signature_stability")


def fig4():
    rows = []
    for f in range(5):
        rows += [r for r in json.load(open(f"{R}/ckpt/fold{f}.json"))["models"] if r["model"] == "KNN"]
    d = pd.concat([pd.DataFrame(rows),
                   pd.read_csv(f"{R}/mlp_keras_equivalent_folds.csv").assign(model="MLP")],
                  ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    for ax, metric, title in [(axes[0], "balanced_accuracy", "Balanced accuracy"),
                              (axes[1], "f1_macro", "Macro F1")]:
        x = 0
        ticks, labels = [], []
        for model in ["KNN", "MLP"]:
            for feats, colr in [("consensus-15", "#0072B2"), ("all-genes", "#999999")]:
                v = d[(d.model == model) & (d.features == feats)][metric].values
                ax.bar(x, v.mean(), color=colr, width=0.7, alpha=0.9)
                ax.scatter(np.full(len(v), x) + np.linspace(-.15, .15, len(v)), v,
                           s=10, color="black", zorder=3)
                ticks.append(x)
                labels.append(f"{model}\n{'15 genes' if feats == 'consensus-15' else 'all genes'}")
                x += 1
            x += 0.5
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylim(0.6, 1.0); ax.set_title(title)
    axes[0].set_ylabel("Score (dots = outer folds)")
    fig.suptitle("Consensus signature vs full transcriptome, nested evaluation",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "Fig4_model_comparison")


def fig5():
    null = np.load(f"{R}/revision/null_5000.npy")
    obs = json.load(open(f"{R}/revision/stats.json"))["null_5000"]["observed"]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.hist(null, bins=40, color="#BBBBBB", edgecolor="white")
    ax.axvline(obs, color="#D55E00", lw=2)
    ax.axvline(np.percentile(null, 95), color="#333333", ls=":", lw=1)
    ax.text(obs - 0.01, ax.get_ylim()[1] * 0.9, f"signature\n{obs:.3f}", color="#D55E00",
            ha="right", fontsize=8)
    ax.text(np.percentile(null, 95) + 0.005, ax.get_ylim()[1] * 0.6, "95th pct", fontsize=7)
    ax.set_xlabel("Macro F1, nested 5-fold CV"); ax.set_ylabel("Random 5-gene sets")
    ax.set_title("Fold-specific top-5 genes vs 5,000 random sets (p < 0.001)",
                 loc="left", fontweight="bold")
    save(fig, "Fig6_null_model")


def fig6():
    X, y, b, _ = load_data(DATA, verbose=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=False)
    rng = np.random.default_rng(0)
    for ax, g in zip(axes, ["PLPPR4", "KRT14", "EMP3"]):
        for i, c in enumerate([1, 2, 3]):
            m = (y.values == c)
            v = X[g].values[m]
            st = b.values[m]
            ax.boxplot(v, positions=[i], widths=0.55, showfliers=False,
                       medianprops=dict(color="black"), boxprops=dict(color="#666666"),
                       whiskerprops=dict(color="#666666"), capprops=dict(color="#666666"))
            ax.scatter(i + rng.uniform(-.2, .2, len(v)), v, s=5, alpha=0.75,
                       c=[STUDY[k] for k in st], lw=0)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Normal", "Primary", "Metastatic"])
        ax.set_title(g, fontweight="bold")
    axes[0].set_ylabel("log2 expression")
    handles = [plt.Line2D([], [], marker="o", ls="", color=STUDY[k], label=f"Study {k+1}")
               for k in range(4)]
    axes[2].legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle("PLPPR4 and EMP3 peak in primary melanoma; colour = source study",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "Fig8_PLPPR4")


def fig7():
    d = pd.read_csv(f"{R}/table4_leave_one_study_out.csv")
    d = d[d.model == "KNN"].sort_values("held_out_study")
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    labels = [f"Study {int(s)+1}\n(n={int(n)}; {'all 3' if len(eval(c))==3 else 'lacks a class'})"
              for s, n, c in zip(d.held_out_study, d.n_test, d.classes_in_test)]
    colr = ["#0072B2" if len(eval(c)) == 3 else "#999999" for c in d.classes_in_test]
    ax.bar(range(len(d)), d.balanced_accuracy, color=colr, width=0.6)
    ax.axhline(0.873, color="#D55E00", ls="--", lw=1)
    ax.text(3.35, 0.885, "pooled nested CV 0.873", color="#D55E00", fontsize=7.5, ha="right")
    for i, v in enumerate(d.balanced_accuracy):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(d))); ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylim(0, 1); ax.set_ylabel("Balanced accuracy (kNN)")
    ax.set_title("Leave-one-study-out: transfer to an unseen cohort", loc="left",
                 fontweight="bold")
    save(fig, "Fig7_leave_one_study_out")


def fig8():
    c = json.load(open(f"{R}/table8_composition.json"))
    by = pd.DataFrame(json.load(open(f"{R}/review/cell_scores_by_class.json")))
    fig = plt.figure(figsize=(6.6, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], width_ratios=[1, 0.04],
                          hspace=0.55, wspace=0.05)
    a = fig.add_subplot(gs[0, 0]); cax = fig.add_subplot(gs[0, 1])
    b = fig.add_subplot(gs[1, :])
    im = a.imshow(by.values, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
    a.set_xticks(range(by.shape[1])); a.set_xticklabels([x.capitalize() for x in by.columns])
    a.set_yticks(range(3)); a.set_yticklabels(["Normal skin", "Primary melanoma", "Metastatic"])
    for i in range(by.shape[0]):
        for j in range(by.shape[1]):
            v = by.values[i, j]
            a.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                   color="white" if abs(v) > 0.7 else "black")
    a.set_title("A  Marker-based cell-type score by class (mean z)", loc="left", fontsize=9,
                fontweight="bold")
    for sp in a.spines.values(): sp.set_visible(False)
    fig.colorbar(im, cax=cax)
    nc = json.load(open(f"{R}/review/part3.json"))
    t3 = {k: {"balanced_accuracy": v} for k, v in nc["three_class"].items()}
    ls = {k: {"balanced_accuracy": v} for k, v in nc["primary_vs_metastatic"].items()}
    keys = [("signature", "Signature, 15 genes"),
            ("adj_keratinocyte", "Signature, adjusted for keratinocyte"),
            ("adj_all", "Signature, adjusted for all 5 types"),
            ("keratinocyte_only", "Keratinocyte score alone"),
            ("all_scores_only", "All 5 cell-type scores alone")]
    yy = np.arange(len(keys))
    v3 = [t3[k]["balanced_accuracy"] for k, _ in keys]
    vl = [ls[k]["balanced_accuracy"] if k in ls else np.nan for k, _ in keys]
    b.barh(yy - 0.19, v3, height=0.36, color="#0072B2", label="Three-class (chance 0.33)")
    b.barh(yy + 0.19, vl, height=0.36, color="#E69F00", label="Primary vs metastatic (chance 0.50)")
    for i, (p, q) in enumerate(zip(v3, vl)):
        b.text(p + 0.008, i - 0.19, f"{p:.2f}", va="center", fontsize=7)
        if not np.isnan(q):
            b.text(q + 0.008, i + 0.19, f"{q:.2f}", va="center", fontsize=7)
    b.set_yticks(yy); b.set_yticklabels([n for _, n in keys]); b.invert_yaxis()
    b.set_xlim(0.3, 1.02); b.set_xlabel("Balanced accuracy, nested 5-fold CV (kNN)")
    b.axhline(2.5, color="#BBBBBB", lw=0.8)
    b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    b.set_title("B  Classification before and after composition adjustment", loc="left",
                fontsize=9, fontweight="bold")
    save(fig, "Fig9_composition")


def fig9():
    cur = json.load(open(f"{R}/roc_out_of_fold.json"))
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for c in ["1", "2", "3"]:
        ax.plot(cur[c]["fpr"], cur[c]["tpr"], color=CLS[int(c)], lw=1.6,
                label=f"{CLSN[int(c)]} (AUC {cur[c]['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="#BBBBBB", ls=":", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    ax.set_title("Out-of-fold ROC, one-vs-rest", loc="left", fontweight="bold")
    ax.set_aspect("equal")
    save(fig, "Fig5_ROC")


def figS1():
    from sklearn.decomposition import PCA
    X, y, b, _ = load_data(DATA, verbose=False)
    Z = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    p = PCA(2, random_state=42).fit(Z.values)
    S = p.transform(Z.values)
    ev = p.explained_variance_ratio_ * 100
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True, sharey=True)
    for c in [1, 2, 3]:
        m = y.values == c
        axes[0].scatter(S[m, 0], S[m, 1], s=8, color=CLS[c], label=CLSN[c], lw=0, alpha=.8)
    for k in range(4):
        m = b.values == k
        axes[1].scatter(S[m, 0], S[m, 1], s=8, color=STUDY[k], label=f"Study {k+1}", lw=0, alpha=.8)
    for ax, t in zip(axes, ["Coloured by disease class", "Coloured by source study"]):
        ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)"); ax.set_title(t, fontsize=9)
        ax.legend(frameon=False, fontsize=7, markerscale=1.5)
    axes[0].set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    fig.suptitle("PCA after joint normalisation: class, not study, drives PC1",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "FigS1_PCA")


def figS2():
    import networkx as nx
    e = pd.read_csv(f"{R}/coexpression_edges.tsv", sep="\t", header=None,
                    names=["a", "b", "w", "s"])
    G = nx.from_pandas_edgelist(e, "a", "b", ["w"])
    bc = nx.betweenness_centrality(G)
    pos = nx.spring_layout(G, seed=42, k=0.45)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.12, width=0.5)
    sizes = [60 + 6000 * bc[n] for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color="#56B4E9",
                           edgecolors="#0072B2", linewidths=0.6)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)
    ax.axis("off")
    ax.set_title("Co-expression network of top-60 consensus genes (|r| ≥ 0.7)\n"
                 "One dense epidermal module; PLPPR4, EMP3, PLOD3, FABP4 have no strong partners",
                 loc="left", fontsize=8.5, fontweight="bold")
    save(fig, "FigS2_coexpression_network")



def figS3():
    from sklearn.decomposition import PCA
    X, y, b, _ = load_data(DATA, verbose=False)
    A = np.load(f"{R}/revision/combat_full_for_pca.npy")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=False, sharey=False)
    for ax, M, t in [(axes[0], X.values, "Before correction"), (axes[1], A, "After ComBat-style location correction")]:
        Z = (M - M.mean(0)) / (M.std(0) + 1e-9)
        p = PCA(2, random_state=42).fit(Z); S = p.transform(Z); ev = p.explained_variance_ratio_ * 100
        for k in range(4):
            m = b.values == k
            ax.scatter(S[m, 0], S[m, 1], s=8, color=STUDY[k], label=f"Study {k+1}", lw=0, alpha=.8)
        ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)"); ax.set_title(t, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7, markerscale=1.5)
    fig.suptitle("Study structure before and after batch correction (coloured by source study)",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "FigS3_batch_correction")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        globals()[name]()
