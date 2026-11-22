"""Typeset the manuscript (text mirrors the Claude Doc at rev 115) with figures."""
import os
import matplotlib
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, KeepTogether, ListFlowable, ListItem,
                                CondPageBreak)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
OUT = os.environ.get("MANUSCRIPT_PDF", "Melanoma_feature_selection_manuscript.pdf")

TTF = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
for name, f in [("Serif", "DejaVuSerif.ttf"), ("Serif-B", "DejaVuSerif-Bold.ttf"),
                ("Serif-I", "DejaVuSerif-Italic.ttf"), ("Serif-BI", "DejaVuSerif-BoldItalic.ttf"),
                ("Sans", "DejaVuSans.ttf"), ("Sans-B", "DejaVuSans-Bold.ttf"),
                ("Sans-I", "DejaVuSans-Oblique.ttf"), ("Sans-BI", "DejaVuSans-BoldOblique.ttf")]:
    pdfmetrics.registerFont(TTFont(name, os.path.join(TTF, f)))
for fam in ("Serif", "Sans"):
    addMapping(fam, 0, 0, fam); addMapping(fam, 1, 0, fam + "-B")
    addMapping(fam, 0, 1, fam + "-I"); addMapping(fam, 1, 1, fam + "-BI")

INK = colors.HexColor("#1A1A1A")
ACCENT = colors.HexColor("#0B4F8A")
NOTE = colors.HexColor("#B00020")
RULE = colors.HexColor("#9AA5B1")

body = ParagraphStyle("body", fontName="Serif", fontSize=9.8, leading=14.2, alignment=TA_LEFT,
                      textColor=INK, spaceAfter=6.5)
title = ParagraphStyle("title", fontName="Serif-B", fontSize=16.5, leading=21, alignment=TA_LEFT,
                       textColor=INK, spaceAfter=10)
meta = ParagraphStyle("meta", parent=body, alignment=TA_LEFT, spaceAfter=3)
h1 = ParagraphStyle("h1", fontName="Sans-B", fontSize=12.5, leading=16, textColor=ACCENT,
                    spaceBefore=12, spaceAfter=6, keepWithNext=True)
h2 = ParagraphStyle("h2", fontName="Sans-B", fontSize=10.6, leading=14, textColor=INK,
                    spaceBefore=8, spaceAfter=4, keepWithNext=True)
cap = ParagraphStyle("cap", fontName="Serif", fontSize=8.4, leading=11.2, alignment=TA_LEFT,
                     textColor=INK, spaceBefore=3, spaceAfter=10)
cell = ParagraphStyle("cell", fontName="Sans", fontSize=7.6, leading=9.6, textColor=INK)
cellh = ParagraphStyle("cellh", parent=cell, fontName="Sans-B")
ref = ParagraphStyle("ref", parent=body, fontSize=8.6, leading=11.6, alignment=TA_LEFT,
                     leftIndent=16, firstLineIndent=-16, spaceAfter=3.2)
bul = ParagraphStyle("bul", parent=body, spaceAfter=2.5)

W = A4[0] - 2 * 2.2 * cm


def note(t):
    return f'<font color="#B00020"><i>[{t}]</i></font>'


from pdf_revisions import fix, USED, REPL


def P(t, st=body):
    return Paragraph(fix(t), st)


def H1(t):
    return Paragraph(t, h1)


def H2(t):
    return Paragraph(t, h2)


def bullets(items, numbered=False):
    return ListFlowable([ListItem(Paragraph(fix(i), bul), leftIndent=14) for i in items],
                        bulletType="1" if numbered else "bullet", start="1" if numbered else None,
                        bulletFontName="Serif", bulletFontSize=9, leftIndent=14,
                        bulletFormat="%s." if numbered else None)


def table(rows, widths, caption, header=True, align_right_from=None):
    data = [[Paragraph(str(c), cellh if (header and i == 0) else cell) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1 if header else 0)
    st = [("LINEABOVE", (0, 0), (-1, 0), 0.9, INK),
          ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
          ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F3F5F8")))
    t.setStyle(TableStyle(st))
    return KeepTogether([t, Paragraph(fix(caption), cap)])


def figure(fname, caption, width=1.0):
    path = os.path.join(FIG, fname)
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    iw = W * width
    img = Image(path, width=iw, height=iw * h / w)
    return KeepTogether([Spacer(1, 4), img, Paragraph(fix(caption), cap)])


def on_page(c, doc):
    c.saveState()
    c.setFont("Sans", 7.5)
    c.setFillColor(colors.HexColor("#5A6470"))
    if doc.page > 1:
        c.drawString(2.2 * cm, A4[1] - 1.35 * cm, "Feature-selection benchmark for melanoma transcriptomics")
        c.drawRightString(A4[0] - 2.2 * cm, A4[1] - 1.35 * cm, "Manuscript draft")
        c.setStrokeColor(RULE); c.setLineWidth(0.4)
        c.line(2.2 * cm, A4[1] - 1.5 * cm, A4[0] - 2.2 * cm, A4[1] - 1.5 * cm)
    c.drawCentredString(A4[0] / 2, 1.2 * cm, str(doc.page))
    c.restoreState()


s = []
# ------------------------------------------------------------------ title page
s += [Spacer(1, 6),
      P("Research article · Manuscript draft for <i>Computers in Biology and Medicine</i>",
        ParagraphStyle("kick", parent=meta, fontName="Sans", fontSize=8.5, textColor=ACCENT)),
      Spacer(1, 6),
      P("Thirty-two feature selectors and their consensus for melanoma progression markers: "
        "a leakage-free assessment of accuracy, stability, tissue composition and cross-cohort transfer", title),
      P("Majid Pilehvar*, Jafar Razmara, Jaber Karimpour", ParagraphStyle("au", parent=meta, fontName="Serif-B", fontSize=10.5)),
      P("Department of Computer Science, Faculty of Mathematics, Statistics and Computer Science, "
        "University of Tabriz, Tabriz, Iran", meta),
      P("*Corresponding author: Majid Pilehvar, majid.pilehvar@gmail.com", meta),
      P("<b>Short title.</b> Leakage-free feature selection for melanoma markers", meta),
      Spacer(1, 8),
      H2("Highlights"),
      bullets(["Leakage-free benchmark of 32 feature selectors on 321 melanoma transcriptomes",
               "Accurate selectors were unstable; the most stable filters were inaccurate",
               "A consensus gave median accuracy; LASSO and Kruskal-Wallis did better",
               "PLPPR4 peaks in primary melanoma and escapes metastasis-vs-normal tests",
               "Tissue composition explains most of the signal; cross-study transfer is weak"]),
      H2("Abstract"),
      P("Feature-selection algorithms applied to high-dimensional transcriptomic data return different gene sets, so "
        "biomarker lists depend on the algorithm as much as on the disease. We assessed, without information leakage, "
        "how 32 selectors and their consensus trade accuracy against stability in melanoma, and how much of the "
        "resulting signal survives tissue-composition adjustment and transfer between cohorts."),
      P("Four public microarray studies were merged into a quantile-normalised matrix of 12,337 genes and 321 samples: "
        "normal skin (31), primary melanoma (153) and metastatic melanoma (137). Selectors from six method families, "
        "and a performance-weighted consensus over them, were fitted inside the training folds of nested cross-validation."),
      P("Accurate selectors, mostly wrappers and metaheuristics, chose different genes in every fold, while the most "
        "stable filters were the least accurate. The consensus reached the median selector's balanced accuracy (0.873) "
        "with above-median stability, but LASSO and Kruskal-Wallis did better on both. Its 15 genes outperformed the "
        "full transcriptome with a dropout network (balanced accuracy 0.907 against 0.848), and its top five beat 500 "
        "random gene sets (p = 0.002). PLPPR4, selected in every fold, peaks in primary melanoma and is missed by "
        "metastasis-versus-normal comparisons. Cell-type composition explained most of the three-class signal: "
        "adjusting for five cell types reduced balanced accuracy from 0.873 to 0.665. Leave-one-study-out balanced "
        "accuracy ranged from 0.46 to 0.86."),
      P("<b>Keywords:</b> melanoma; biomarker discovery; feature selection; selection stability; consensus ranking; "
        "gene expression; cross-cohort validation"),
      PageBreak()]

# ------------------------------------------------------------------ 1 introduction
s += [H1("1. Introduction"),
      P("The gene sets returned by feature selection on transcriptomic data depend heavily on the algorithm. In the "
        "benchmark reported here, the median pair of 32 methods applied to the same 12,337 genes and 321 samples shared "
        "two of fifteen selected genes, and 72 of 496 pairs shared none. Any single-method biomarker study therefore "
        "reports one draw from a wide distribution of possible answers."),
      P("Melanoma makes this costly. It accounts for a small share of skin cancer diagnoses and a large share of skin "
        "cancer deaths [1], because it metastasises early and the window in which treatment changes outcome is narrow. "
        "Transcriptomic markers that track the progression from normal skin through primary tumour to metastasis would "
        "support earlier risk stratification, and public repositories now hold enough melanoma expression data to look "
        "for them. What is missing is not data but a selection procedure whose output does not change when the "
        "algorithm does."),
      P("Prior computational work on melanoma biomarkers has largely followed one of two patterns, and each has a "
        "characteristic failure. The first is differential expression with a significance threshold, which finds too "
        "much: in our merged cohort, 6,600 of 12,337 genes differ between metastatic and normal samples at a false "
        "discovery rate below 0.05, so significance alone ranks nothing. The second is a single machine-learning "
        "selector paired with a classifier, which finds too arbitrarily. Ein-Dor et al. [2] showed that breast-cancer "
        "outcome signatures derived this way are not unique, and that many disjoint gene sets achieve comparable "
        "accuracy on the same data; Michiels et al. [3] reached the same conclusion by repeated random validation."),
      P("This paper makes five contributions."),
      bullets(["<b>A leakage-free benchmark of 32 selectors.</b> Every selector is fitted inside training folds only and "
               "scored on accuracy, fold-to-fold stability and runtime. Selecting on the full matrix instead inflated "
               "accuracy for 24 of the 32 methods, by up to 5.9 percentage points.",
               "<b>An accuracy-stability map.</b> The most accurate selectors chose largely different genes in every "
               "fold, while the most stable were the least accurate; LASSO, Kruskal-Wallis and stability selection "
               "were best on both counts.",
               "<b>A tested consensus.</b> A performance-weighted, redundancy-aware consensus is evaluated under the "
               "same nesting. It matched the median selector, and an ablation shows that its weighting and redundancy "
               "control made little difference, a negative result for elaborate ensembles.",
               "<b>A composition-aware reading of the signature.</b> Marker-based cell-type scores and in-fold "
               "residualisation separate genes that track epidermal content from genes whose signal survives adjustment.",
               "<b>An explicit test of cross-cohort transfer.</b> Holding out each study in turn shows how much pooled "
               "cross-validation overstates generalisation."], numbered=True),
      Spacer(1, 4),
      P("The analysis also highlights a gene that a metastasis-versus-normal contrast discards. PLPPR4 was chosen by 23 "
        "of the 32 selectors and ranked first in the consensus in all five folds, yet shows no difference between "
        "metastatic and normal samples (FDR = 0.46), because its expression is non-monotonic: elevated in primary "
        "melanoma relative to both normal skin and metastases. Comparing metastases with normal tissue, the two ends of "
        "progression, cannot see this pattern; any analysis that models all three stages can."),
      H2("1.1 Related work"),
      P("<b>Feature selection in bioinformatics.</b> Saeys et al. [4] established the now-standard taxonomy of filter, "
        "wrapper and embedded methods for high-dimensional biological data, and documented that no family dominates "
        "across datasets. Guyon et al. [5] introduced recursive feature elimination for gene selection with support "
        "vector machines, still among the most widely applied embedded approaches."),
      P("<b>Stability as a criterion.</b> That different selectors return different gene sets from the same data was "
        "framed as a measurable property rather than a nuisance by Saeys et al. [6], who proposed combining multiple "
        "selection runs to improve robustness. Abeel et al. [7] extended this to cancer biomarker identification, "
        "showing that ensembles built from resampled data substantially increase the stability of selected markers "
        "while maintaining or improving classification accuracy. Our approach differs in two respects: we ensemble "
        "across <i>different algorithms</i> rather than across resamples of one algorithm, and we weight each "
        "algorithm's contribution by its measured discriminative performance rather than counting votes equally. "
        "Bolón-Canedo and Alonso-Betanzos review the wider design space of such ensembles [8]."),
      P("<b>Selection bias.</b> Ambroise and McLachlan [9] demonstrated that cross-validation performed after gene "
        "selection on the full dataset yields near-zero error rates that vanish once selection is moved inside the "
        "cross-validation loop. Simon et al. [10] made the same argument for diagnostic and prognostic classifiers. "
        "Despite being two decades old, this error remains common in applied biomarker papers, and Section 3.1 "
        "quantifies its magnitude on the present data."),
      P("<b>Melanoma markers.</b> PRAME has become a widely used immunohistochemical marker for distinguishing melanoma "
        "from benign naevi [11]. Maspin (SERPINB5) loss has been associated with melanoma invasion and progression in "
        "several independent series [12,13,14]. S100A7 (psoriasin) is upregulated in epithelial skin tumours [15] and "
        "proposed as a progression marker across carcinomas [16]. None of these was derived by a consensus procedure, "
        "and Section 3.4 compares their single-gene discriminative power against the genes our framework selects.")]

# ------------------------------------------------------------------ 2 methods
s += [H1("2. Materials and methods"),
      H2("2.1 Data and preprocessing"),
      P("Four public melanoma microarray series were retrieved from the Gene Expression Omnibus [17]. After joint "
        "quantile normalisation [18] across all samples, the working matrix held 12,337 genes and 321 samples in three "
        "classes defined by the source annotations: normal skin (n = 31), primary cutaneous melanoma (n = 153) and "
        "metastatic melanoma (n = 137)."),
      table([["Study", "GEO series", "Platform", "Normal skin", "Primary", "Metastatic", "Source"],
             ["1", "GSE3189", "GPL96", "7", "45", "0", "Talantov et al. [19]"],
             ["2", "GSE8401", "GPL96", "0", "31", "52", "Xu et al. [20]"],
             ["3", "GSE15605", "GPL570", "16", "46", "12", "Raskin et al. [21]"],
             ["4", "GSE46517", "GPL96", "8", "31", "73", "Kabbarah et al. [22]"]],
            [0.08, 0.14, 0.12, 0.13, 0.11, 0.13, 0.29],
            "<i>Table S1. Composition of the merged cohort. Benign naevi present in GSE3189 (18) and GSE46517 (9) were "
            "not part of the analysed matrix, so no class contains naevi. Two studies lack a class, which matters for "
            "the cross-cohort analysis.</i>"),
      P("<b>Construction of the merged matrix.</b> The matrix was assembled during the first author's MSc work, and its "
        "processing steps were reconstructed and verified against the public GEO records. The four series were deposited "
        "with different processing (GSE3189: MAS5; GSE8401 and GSE46517: RMA; GSE15605: log2 MAS5 after quantile "
        "normalisation, probe-filtered). Expression was summarised per gene by averaging all probe sets annotated to a gene "
        "on the log2 scale: for 200 randomly chosen genes, this rule reproduced the within-sample gene order of the merged "
        "matrix far better than keeping one probe per gene (median Spearman ρ 0.96 to 0.98 in GSE3189, GSE8401 and "
        "GSE46517, against 0.86 to 0.88 for the highest-mean probe; Table S6). Agreement for single-probe genes was 0.95 to "
        "0.97 rather than 1, which quantile normalisation alone would give, and was lowest for GSE15605 (0.88), whose "
        "deposited values are filtered MAS5 data; the arrays were therefore re-summarised from the raw CEL files before "
        "merging, consistent with a uniform RMA-type summarisation rather than the heterogeneous deposited values. Gene "
        "symbols follow the Affymetrix annotation of the time (11,469 of 12,339 match current GPL96 symbols), and the gene "
        "set corresponds to the genes on the HG-U133A array shared by both platforms. The merged matrix was then "
        "quantile-normalised across all 321 samples, which the identical per-sample quantiles of the matrix confirm. "
        + note("Authors: confirm the summarisation method (RMA, and the R/Bioconductor package used) from the original MSc scripts")),
      P("Two cleaning steps were applied. Eighteen rows carried gene symbols that spreadsheet software had converted to "
        "dates; they were restored to MARCHF and SEPTIN symbols (Mar-01 to MARCHF1, Sep-02 to SEPTIN2, and so on). Two "
        "date strings occurred twice, most likely because MARCHF1 and MTARC1, and MARCHF2 and MTARC2, convert to the "
        "same dates; each pair was averaged, and none of these genes ranks among the 300 highest consensus genes, so "
        "the choice does not affect any result. Obsolete symbols were updated to current HGNC nomenclature, including "
        "LPPR4 to PLPPR4."),
      P("Source study was inferred from GSM accession blocks and retained as a covariate for the validation analyses in "
        "Section 2.5. Residual batch structure was assessed two ways. In principal component analysis (Fig. S1), the "
        "first component's variance was attributable 38% to disease class and 1% to source study. A random forest "
        "nonetheless predicted source study at 0.65 accuracy against a 0.35 majority baseline, so study effects are "
        "reduced but not eliminated by the joint normalisation."),
      H2("2.2 Feature selection methods"),
      P("Thirty-two algorithms were implemented behind a single interface, <font name='Sans'>fn(X, y, k, rng)</font>, "
        "returning k = 15 selected column indices:"),
      table([["Family", "Methods"],
             ["Filter", "ANOVA, chi-square, mutual information, information gain, Fisher score, Pearson correlation, "
                        "Spearman correlation, Kruskal-Wallis, variance threshold"],
             ["Embedded", "LASSO, elastic net, ridge, random forest importance, extra trees importance, permutation "
                          "importance, stability selection, Boruta"],
             ["Projection", "PCA loading, LDA coefficient, CCA"],
             ["Information-theoretic", "mRMR, MIFS"],
             ["Wrapper", "RFE, forward selection, backward selection, beam search"],
             ["Metaheuristic", "genetic algorithm, PSO, ABC, ACO, differential evolution, simulated annealing"]],
            [0.2, 0.8], ""),
      P("Every stochastic method received an explicit seed. Methods whose cost grows steeply with the number of genes "
        "first shortlisted candidates by ANOVA F computed <b>within the same training fold</b>, which keeps the "
        "procedure leak-free while remaining tractable: 3,000 genes for mutual information and information gain; 1,000 "
        "for LASSO, elastic net and stability selection; 500 for Boruta; 300 for permutation importance, LDA, CCA, "
        "mRMR, MIFS, RFE and the metaheuristics; and 60 to 150 for backward, beam and forward search. Population-based "
        "methods used 20 agents over 15 generations, with 3-fold internal cross-validation as the fitness function."),
      P("Key method references: LASSO [23], random forest importance [24], stability selection [25], Boruta [26], "
        "mRMR [27], MIFS [28] and RFE [5]."),
      H2("2.3 Consensus scoring"),
      P("For each training fold (Fig. 1), every method m produced a selected set S<sub>m</sub> and an accuracy "
        "a<sub>m</sub> estimated by 3-fold cross-validation <b>on the training data only</b>, so no held-out information "
        "entered the weights. Methods whose selected sets overlapped above a Jaccard coefficient of 0.80 were clustered "
        "by union-find and assigned weight w<sub>m</sub> = 1 / (cluster size), so near-duplicate methods share a single "
        "vote rather than each casting one. The consensus score of gene g was then the sum of w<sub>m</sub> × "
        "a<sub>m</sub> over all methods that selected g. The 15 highest-scoring genes formed the fold's signature."),
      figure("Fig1_pipeline.png", "<b>Fig. 1.</b> Analysis pipeline. All selection, weighting, redundancy clustering and "
             "consensus steps run inside each outer training fold; held-out samples are used only for evaluation.", 0.92),
      H2("2.4 Evaluation"),
      P("Stratified 5-fold cross-validation with shuffling (random_state = 42) was used throughout. Shuffling matters "
        "here because samples in the source matrix are ordered by class and then by study, so unshuffled folds would "
        "be contiguous blocks of a single cohort."),
      P("The full selection and consensus procedure was repeated inside each outer fold, and the resulting signature "
        "evaluated on that fold's held-out samples, with two classifiers: k-nearest neighbours (k = 5) and a multilayer "
        "perceptron matching the original Keras model (150-50-20 ReLU units, dropout 0.5, Adam, batch size 8, up to 500 "
        "epochs with early stopping on a stratified 15% split of the training fold). Both were also trained on all "
        "12,337 genes as a baseline."),
      P("Reported metrics are accuracy, balanced accuracy, macro precision, macro recall reported once as sensitivity, "
        "macro F1, one-vs-rest specificity averaged across the three classes, and one-vs-rest macro AUC."),
      P("<b>Stability and selection bias.</b> A method's fold-to-fold stability is the mean pairwise Jaccard index "
        "between the gene sets it selected in the five outer training folds. To measure selection bias, each method's "
        "selection on the full matrix was also evaluated by the same cross-validation, the arrangement common in the "
        "literature, and compared with its nested estimate."),
      P("<b>Consensus ablation.</b> In each fold the consensus was recomputed with performance weights and redundancy "
        "weights, with either one alone, and with neither (a plain vote count), using the same per-method selections."),
      H2("2.5 Validation analyses"),
      P("<b>Cross-cohort transfer.</b> Each source study was held out in turn while the entire consensus pipeline was "
        "re-derived from the remaining three."),
      P("<b>Differential expression.</b> Welch t-tests for three contrasts (primary vs normal, metastatic vs normal, "
        "metastatic vs primary) across all 12,337 genes, with Benjamini-Hochberg correction [29]."),
      P("<b>Comparison with published markers.</b> In each outer fold, the 30 highest consensus genes from that fold's "
        "training data and 14 previously proposed melanoma markers (PRAME, PMEL, TYRP1, TYR, MLANA, MITF, SOX10, S100B, "
        "DCT, MAGEA1, MAGEA3, CDKN2A, BRAF, NRAS) were each used alone in a random forest trained on the training part "
        "and scored on the held-out part. The groups were compared with a one-sided Mann-Whitney U test over gene-fold "
        "scores, and fold by fold."),
      P("<b>Null model.</b> The five highest consensus genes of each outer fold were evaluated on that fold's held-out "
        "samples by kNN macro F1. The same was done for 500 random five-gene sets, giving an empirical p-value with the "
        "usual +1 correction."),
      P("<b>Tissue composition.</b> Bulk tissue mixes tumour cells with epidermis, stroma, vessels and immune cells, and "
        "metastases are often sampled from lymph nodes or distant sites that lack epidermis. We therefore scored each "
        "sample for five cell types as the mean z-score of canonical markers: keratinocyte (KRT10, DSG1, IVL, TP63, "
        "PERP, SPRR1A), melanocyte (PMEL, MLANA, DCT, MITF, SOX10, S100B), fibroblast (COL1A1, COL1A2, COL3A1, DCN, "
        "LUM, PDGFRB, FBLN1), endothelial (PECAM1, VWF, CDH5, CLDN5, EMCN) and immune (PTPRC, CD3E, CD2, MS4A1, CD14, "
        "CD68, LYZ, CD53). Genes in any fold's signature or among the top 60 consensus genes were excluded from the "
        "marker sets, so that adjusting the signature for composition does not adjust it for itself. In each outer "
        "fold, the fold's own 15 signature genes were regressed on the composition scores using training samples only, "
        "and classification was repeated on the residuals. This marker-based score is a transparent proxy for "
        "reference-based deconvolution such as CIBERSORTx [30] or xCell [31], not a substitute for it."),
      P("<b>Co-expression network.</b> Pairwise Pearson correlations among the top 60 consensus genes were thresholded "
        "at |r| ≥ 0.7, and betweenness centrality was computed with NetworkX."),
      P("<b>Functional enrichment and protein interactions.</b> Code for GO Biological Process and KEGG enrichment of "
        "the top 60 genes against the 12,337-gene array background, and for a STRING [32] network at combined score "
        "≥ 700, is included in the release (enrichment_ppi.py). These analyses need internet access and are not part "
        "of the results reported here."),
      H2("2.6 Implementation"),
      P("Python 3.12 with scikit-learn [33], NumPy, SciPy, pandas and NetworkX. In the main analysis, MIC [34] is not "
        "among the 32 selectors, and Boruta [26] runs a fixed number of shadow-feature rounds without its sequential "
        "tests. The multilayer perceptron is a NumPy re-implementation of the Keras network with Glorot "
        "initialisation, dropout and Adam as in Keras. Section 3.9 repeats the analysis with the full Boruta algorithm "
        "(two-step binomial testing with Benjamini-Hochberg and Bonferroni correction, as in boruta_py) and with MIC "
        "computed exactly for a categorical target (B(n) = n<super>0.6</super>; optimal partitions of the gene axis by "
        "dynamic programming over class-boundary points, verified against exhaustive search). The released code also "
        "switches to minepy, boruta_py and TensorFlow automatically when those packages are installed.")]

# ------------------------------------------------------------------ 3 results
s += [H2("2.7 Sensitivity analyses"),
      P("<b>Uncertainty.</b> Fold-level scores are summarised as mean ± standard deviation with t-based 95% confidence "
        "intervals over the five outer folds. Differences between the consensus and individual selectors were tested with "
        "the corrected resampled t-test of Nadeau and Bengio [35], which accounts for overlapping training sets, with Holm "
        "correction across comparisons [36]."),
      P("<b>Classifier dependence.</b> Each selector's fold-wise genes and the consensus genes were re-evaluated with "
        "L2-regularised logistic regression and a 300-tree random forest in addition to kNN."),
      P("<b>Batch correction.</b> A ComBat-style empirical-Bayes location adjustment [37], with disease class as a "
        "protected covariate, was estimated on the training samples of each fold and applied to held-out samples using "
        "only their study label, never their class. On simulated data with the same study-by-class design and additive "
        "study effects, this procedure reduced study identifiability from 1.00 to 0.37 (chance 0.35). In "
        "leave-one-study-out validation the held-out study has no labelled training samples, so each study was "
        "mean-centred on its own as the only label-free alternative."),
      P("<b>Repeated samples.</b> Pairwise sample correlations were screened for near-duplicates (r ≥ 0.98), which "
        "would indicate repeated arrays or paired tissue from one patient."),
      P("<b>External validation.</b> Expression of the 15 full-data signature genes in TCGA-SKCM [39] (RNA-seq, RSEM; "
        "103 primary and 369 metastatic tumours) was retrieved from cBioPortal [40]. A kNN and a logistic-regression model "
        "were trained on the microarray primary and metastatic samples only and applied unchanged to TCGA, after z-scoring "
        "each platform within itself (log2(RSEM + 1) for TCGA). Nothing was selected or tuned on TCGA, and TCGA contains "
        "almost no normal skin, so only the primary-versus-metastatic task could be tested."),
      P("<b>Established composition method.</b> The composition adjustment was repeated with MCP-counter [38], "
        "re-implemented from its published marker table (each population scored as the mean log2 expression of its "
        "markers, as in the original package); 94 of its 111 markers were present on the arrays. Because MCP-counter "
        "covers immune and stromal populations only, it was also combined with the keratinocyte and melanocyte scores "
        "of Section 2.5."),
      P("<b>Repeated cross-validation.</b> The full nested pipeline was repeated on two further random partitions "
        "(seeds 7 and 2026), giving 15 outer folds; comparisons across them used the corrected test with variance "
        "factor 1/15 + 0.25.")]

s += [H1("3. Results"),
      H2("3.1 Accuracy, stability and selection bias of 32 selectors"),
      P("Under leak-free nested evaluation, balanced accuracy across the 32 methods ranged from 0.731 to 0.910 "
        "(Fig. 2). Metaheuristic and information-theoretic methods led; simple correlation filters trailed. Runtime "
        "varied by three orders of magnitude without tracking accuracy: ACO reached the best balanced accuracy in "
        "2.9 s, while beam search took 26.5 s for a lower score."),
      table([["Method", "Family", "Accuracy", "Balanced acc.", "Macro F1", "AUC", "Stability", "Runtime (s)"],
             ["ACO", "metaheuristic", "0.913", "0.910", "0.910", "0.969", "0.04", "2.9"],
             ["MIFS", "information", "0.894", "0.905", "0.901", "0.963", "0.13", "0.6"],
             ["Forward selection", "wrapper", "0.894", "0.896", "0.898", "0.968", "0.18", "18.2"],
             ["Simulated annealing", "metaheuristic", "0.878", "0.893", "0.888", "0.961", "0.05", "1.8"],
             ["Boruta", "embedded", "0.891", "0.893", "0.894", "0.959", "0.11", "4.8"],
             ["Genetic algorithm", "metaheuristic", "0.900", "0.891", "0.899", "0.967", "0.05", "1.5"],
             ["Kruskal-Wallis", "filter", "0.885", "0.889", "0.882", "0.957", "0.34", "5.5"],
             ["LASSO", "embedded", "0.882", "0.886", "0.890", "0.960", "0.34", "0.1"],
             ["Stability selection", "embedded", "0.888", "0.882", "0.893", "0.958", "0.51", "0.8"],
             ["ANOVA", "filter", "0.838", "0.826", "0.834", "0.943", "0.33", "0.1"],
             ["Variance threshold", "filter", "0.797", "0.782", "0.792", "0.912", "0.93", "0.0"],
             ["Spearman correlation", "filter", "0.763", "0.731", "0.731", "0.876", "0.35", "1.2"]],
            [0.2, 0.14, 0.1, 0.12, 0.1, 0.08, 0.1, 0.11],
            "<i>Table 1 (abridged; all 32 rows in table1_method_benchmark.csv). Means over 5 outer folds, kNN on each "
            "method's 15 selected genes. Stability: mean pairwise Jaccard index of the gene sets selected in the five "
            "outer training folds.</i>"),
      figure("Fig2_method_benchmark.png", "<b>Fig. 2.</b> Balanced accuracy (left) and runtime per fold on a log scale "
             "(right) for 32 feature-selection methods under nested 5-fold cross-validation, coloured by method family. "
             "Dotted line: kNN on all 12,337 genes.", 0.9),
      P("Redundancy clustering merged ANOVA, Fisher score and permutation importance into one voting unit, and variance "
        "threshold with PCA loading into another, on measured output overlap rather than on family labels."),
      P("Accuracy and stability pulled in opposite directions (Fig. 3A). The five most accurate selectors, all wrappers, "
        "metaheuristics, Boruta or MIFS, chose largely different genes in every fold (mean pairwise Jaccard 0.04 to "
        "0.18), whereas the most stable, PCA loading, variance threshold and chi-square (Jaccard 0.88 to 0.95), were "
        "among the least accurate. Three selectors combined both properties: Kruskal-Wallis (balanced accuracy 0.889, "
        "Jaccard 0.34), LASSO (0.886, 0.34) and stability selection (0.882, 0.51). Across all pairs of methods, the "
        "median overlap of full-data selections was two of fifteen genes, and 72 of 496 pairs shared none."),
      P("Selecting genes on the full matrix before cross-validation, instead of inside each fold, inflated accuracy for "
        "24 of the 32 methods, by a median of 1.6 and at most 5.9 percentage points (ANOVA: 0.897 against 0.838). The "
        "bias is real, but with a signal this strong it is smaller than the differences between methods."),
      figure("Fig3_signature_stability.png", "<b>Fig. 3.</b> (A) Balanced accuracy against fold-to-fold stability (mean "
             "pairwise Jaccard index of the genes selected in the five outer training folds) for 32 selectors, coloured "
             "by family; star: consensus. (B) Genes in the consensus signature of at least two of the five outer folds."),
      H2("3.2 The consensus signature and its stability"),
      P("Three genes were selected in all five independent folds (Fig. 3B): <b>PLPPR4, S100A7, KRT14</b>. Four more "
        "appeared in four of five: KRT6A, KRT15, KRT16, SERPINB5. On the full-data discovery run the ten highest-scoring "
        "genes were PLPPR4, KRT6A, KRT14, EMP3, ANXA8, KRT16, S100A7, SERPINB5, KLF5 and AHNAK."),
      table([["Gene", "Folds selected (of 5)", "Consensus score", "log2FC metastatic vs normal", "FDR"],
             ["PLPPR4", "5", "18.60", "-0.17", "0.462"], ["S100A7", "5", "5.88", "-4.44", "< 0.001"],
             ["KRT14", "5", "9.28", "-6.72", "< 0.001"], ["KRT6A", "4", "10.75", "-4.89", "< 0.001"],
             ["KRT15", "4", "3.08", "-6.51", "< 0.001"], ["SERPINB5", "4", "5.65", "-5.47", "< 0.001"],
             ["KRT16", "4", "6.69", "-3.79", "< 0.001"], ["EMP3", "3", "7.12", "+1.00", "< 0.001"],
             ["TSPAN8", "3", "3.55", "-2.11", "< 0.001"], ["AHNAK", "3", "5.05", "-2.15", "< 0.001"]],
            [0.16, 0.2, 0.18, 0.3, 0.16],
            "<i>Table 2. Signature stability across outer folds, with differential expression for context.</i>"),
      P("The consensus itself sat near the middle of the accuracy-stability map. Its balanced accuracy (0.873) equalled "
        "the median selector's and its stability (Jaccard 0.29) exceeded the median (0.20), but Kruskal-Wallis, LASSO "
        "and stability selection were better on both counts. The ablation in Table S2 shows that the consensus "
        "refinements did not change this: weighting votes by accuracy, sharing votes among redundant methods, both, or "
        "neither gave balanced accuracy between 0.858 and 0.873 and three or four genes in every fold."),
      table([["Consensus variant", "Accuracy", "Balanced acc.", "Genes in all 5 folds", "Mean pairwise Jaccard"],
             ["Accuracy weights + redundancy weights (main)", "0.888", "0.873", "3", "0.287"],
             ["Accuracy weights only", "0.891", "0.858", "3", "0.301"],
             ["Redundancy weights only", "0.885", "0.871", "4", "0.308"],
             ["Plain vote count", "0.894", "0.860", "3", "0.307"]],
            [0.4, 0.12, 0.14, 0.16, 0.18],
            "<i>Table S2. Consensus ablation under nested 5-fold cross-validation, kNN on each fold's top 15 genes.</i>"),
      H2("3.3 Selected genes outperform the full matrix"),
      table([["Model", "Features", "Accuracy", "Balanced acc.", "Macro F1", "Specificity", "AUC"],
             ["kNN", "consensus-15", "0.888", "0.873", "0.885", "0.932", "0.957"],
             ["kNN", "all 12,337 genes", "0.850", "0.859", "0.859", "0.909", "0.962"],
             ["MLP (dropout)", "consensus-15", "0.897", "0.907", "0.906", "0.939", "0.951"],
             ["MLP (dropout)", "all 12,337 genes", "0.850", "0.848", "0.835", "0.914", "0.947"]],
            [0.17, 0.19, 0.11, 0.14, 0.12, 0.14, 0.11],
            "<i>Table 3. Nested evaluation: the consensus is re-derived inside each fold, so these numbers are not "
            "circular.</i>"),
      P("The signature outperformed the full transcriptome with both classifiers (Fig. 4), and the gain was larger for "
        "the network (balanced accuracy 0.907 against 0.848) than for kNN (0.873 against 0.859). Regularisation decides "
        "this comparison: a scikit-learn perceptron with the same layers but no dropout did worse on 15 genes than on "
        "all genes (0.772 against 0.833), so the advantage of a compact signature for neural classifiers should not be "
        "assumed without it."),
      P("Out-of-fold one-vs-rest AUCs for the kNN consensus model were 0.948 for normal skin, 0.955 for primary "
        "melanoma and 0.965 for metastatic melanoma (Fig. 5)."),
      figure("Fig4_model_comparison.png", "<b>Fig. 4.</b> Consensus 15-gene signature against all genes for kNN and the "
             "dropout network (MLP). Bars: mean over outer folds; dots: individual folds.", 0.85),
      figure("Fig5_ROC.png", "<b>Fig. 5.</b> Out-of-fold one-vs-rest ROC curves for the nested kNN consensus model.", 0.48),
      H2("3.4 The signature is not a chance finding"),
      P("Each fold's five highest consensus genes, evaluated on that fold's held-out samples, reached macro F1 0.843, "
        "whereas 500 random five-gene sets had mean 0.519, 95th percentile 0.650 and maximum 0.722 (Fig. 6). No random "
        "set matched the signature (empirical p = 0.002, the smallest attainable with 500 draws)."),
      P("Used one at a time, the 30 highest consensus genes of each fold achieved mean macro F1 0.485 on held-out "
        "samples, against 0.441 for 14 previously published melanoma markers (one-sided Mann-Whitney U over gene-fold "
        "scores, p = 0.0004). The proposed genes scored higher in each of the five folds."),
      figure("Fig6_null_model.png", "<b>Fig. 6.</b> Macro F1 of each fold's five highest consensus genes on that fold's "
             "held-out samples (orange) against 500 random five-gene sets evaluated on the same folds; dotted line: "
             "95th percentile of the null distribution.", 0.62),
      H2("3.5 Cross-cohort transfer is limited"),
      table([["Held-out study", "n", "Classes present", "kNN balanced acc.", "kNN macro F1"],
             ["Study 1", "52", "normal, primary", "0.457", "0.385"],
             ["Study 2", "83", "primary, metastatic", "0.859", "0.579"],
             ["Study 3", "74", "all three", "0.660", "0.585"],
             ["Study 4", "112", "all three", "0.848", "0.842"]],
            [0.2, 0.1, 0.3, 0.2, 0.2],
            "<i>Table 4. Leave-one-study-out: the whole pipeline is re-derived from the other three cohorts.</i>"),
      P("Performance that reaches 0.873 balanced accuracy under pooled cross-validation falls to between 0.46 and 0.86 "
        "when an entire cohort is held out (Fig. 7). The worst case, Study 1, contains only normal skin and primary "
        "tumours, and the model trained on the other three studies classified 36 of its 45 primary melanomas as "
        "metastatic. Missing classes do not explain the gap between the two complete studies (0.660 and 0.848). This "
        "is the most consequential of the paper's negative results."),
      figure("Fig7_leave_one_study_out.png", "<b>Fig. 7.</b> Leave-one-study-out balanced accuracy (kNN). Grey bars: "
             "held-out studies lacking a class. Dashed line: pooled nested cross-validation.", 0.66),
      H2("3.6 PLPPR4: invisible when metastases are compared with normal skin"),
      P("PLPPR4 ranked first in every fold and carries the highest consensus score by a wide margin, yet its "
        "metastatic-versus-normal contrast is not significant (log2FC -0.17, FDR 0.46). Its class means explain why: "
        "6.69 in normal skin, <b>8.34 in primary melanoma</b>, 6.52 in metastases (Fig. 8). Both contrasts involving "
        "primary tumours are highly significant (primary vs normal FDR 8.2 × 10<super>−10</super>; metastatic vs primary "
        "FDR 2.8 × 10<super>−27</super>), so only the metastasis-versus-normal comparison misses it; pooling all tumours "
        "against normal skin detects it (p = 1 × 10<super>−4</super>), at about half the effect size seen in primary "
        "tumours."),
      P("The pattern is not a batch artefact: the standard deviation of PLPPR4's study means is 0.17 log2 units, "
        "against an average within-study standard deviation of 1.49. EMP3, selected in three of five folds, follows the "
        "same shape (9.63, 11.44 and 10.63 across the three classes). PLPPR4 (phospholipid phosphatase-related protein "
        "4) belongs to the plasticity-related gene family; its transient rise in primary tumours with return to baseline "
        "in metastases is a pattern worth experimental follow-up."),
      figure("Fig8_PLPPR4.png", "<b>Fig. 8.</b> Expression of PLPPR4, KRT14 and EMP3 by class, with points coloured by "
             "source study. PLPPR4 and EMP3 peak in primary melanoma; KRT14 declines monotonically."),
      H2("3.7 Tissue composition explains much, but not all, of the signal"),
      P("Marker-based cell-type scores differed sharply between classes (Fig. 9A). Normal skin scored highest for "
        "keratinocyte (+0.92), fibroblast and endothelial content and lowest for melanocyte content (-1.16); metastases "
        "had the lowest keratinocyte score (-0.58), consistent with metastatic deposits that contain little or no "
        "epidermis."),
      table([["Features", "Three-class balanced acc.", "Primary vs metastatic balanced acc."],
             ["Signature, 15 genes", "0.873", "0.899"],
             ["Signature, adjusted for keratinocyte score", "0.864", "0.822"],
             ["Signature, adjusted for all five scores", "0.665", "0.796"],
             ["Keratinocyte score alone", "0.481", "0.655"],
             ["All five cell-type scores alone", "0.874", "0.855"]],
            [0.46, 0.24, 0.3],
            "<i>Table 5. Composition adjustment within nested cross-validation: each fold's own 15 signature genes, "
            "residuals computed from training samples only, kNN classifier. Chance is 0.33 for three classes and 0.50 "
            "for two.</i>"),
      P("Keratinocyte content alone classifies poorly, and adjusting for it costs the signature about one point. Five "
        "composition scores together, however, classify as well as the signature (0.874 against 0.873), and adjusting "
        "for all five reduces three-class balanced accuracy to 0.665. Composition therefore carries most, though not "
        "all, of the three-class signal. Between primary and metastatic tumours, where both samples are lesional, 0.796 "
        "survives full adjustment against a chance level of 0.50."),
      P("Gene by gene, the keratins, S100A7, SERPINB5, JUP and ANXA8 correlate with keratinocyte score at r between "
        "0.90 and 0.96. PLPPR4 correlates moderately (r = 0.40), and EMP3 (r = 0.06), FABP4 (r = 0.04) and CDC42EP3 "
        "(r = -0.10) hardly at all; EMP3 correlates instead with melanocytic content (r = 0.52), consistent with "
        "expression in tumour cells."),
      figure("Fig9_composition.png", "<b>Fig. 9.</b> Tissue composition. (A) Mean marker-based cell-type score by class. "
             "(B) Balanced accuracy before and after in-fold adjustment for composition, for the three-class task and "
             "for primary versus metastatic melanoma.", 0.82),
      H2("3.8 Co-expression structure of the top 60 genes"),
      P("At |r| ≥ 0.7, 41 of the top 60 consensus genes formed a network of 574 edges in two components (Fig. S2). A "
        "38-gene module contained the keratins, SERPINB5, S100A7, KLF5, JUP and other epidermal genes, nearly all "
        "connected to each other (median degree 32). A three-gene module of KRT19, PIP and SCGB1D2 points to adnexal "
        "gland content. Nineteen genes had no strong partner, among them PLPPR4, EMP3, FABP4, PLOD3, CDC42EP3, TYR and "
        "TYRP1. The dense epidermal module is the network view of the composition effect in Section 3.7; the isolated "
        "genes carry signal the module does not."),
      H2("3.9 Robustness to implementation choices"),
      P("Replacing the simplified Boruta with the full algorithm and adding exact MIC as a 33rd selector left the "
        "results essentially unchanged (Table S3). The other 31 selectors' fold-wise selections were reused unchanged, "
        "so the comparison isolates the two substitutions."),
      table([["Quantity", "Main analysis", "Reference configuration"],
             ["Genes shared with main fold signature (of 15), folds 1-5", "", "13, 13, 13, 14, 13"],
             ["Genes selected in all five folds", "KRT14, PLPPR4, S100A7", "KRT14, PLPPR4, S100A7"],
             ["Full-data top 15 / top 60 shared with main ranking", "", "14 / 56"],
             ["kNN consensus-15, accuracy / balanced acc. / macro F1", "0.888 / 0.873 / 0.885", "0.888 / 0.873 / 0.875"],
             ["Boruta, balanced accuracy", "0.893 (simplified)", "0.882 (full algorithm)"],
             ["MIC, balanced accuracy", "not included", "0.861 (exact)"]],
            [0.46, 0.26, 0.28],
            "<i>Table S3. Main analysis against the reference configuration (full Boruta, exact MIC). Nested 5-fold "
            "cross-validation throughout.</i>"),
      P("The one change in the full-data top 15 swapped JUP for BBOX1, both members of the epidermal module. Exact MIC "
        "ranked in the middle of the 33 selectors, close to mutual information and information gain, as expected for "
        "estimators of the same quantity.")]

# ------------------------------------------------------------------ 4 discussion
s += [H2("3.10 Uncertainty, classifier dependence, batch effects and repeated samples"),
      P("Fold-to-fold variation was substantial. The consensus reached balanced accuracy 0.873 ± 0.066 (95% CI 0.79 to "
        "0.96), with stability 0.29 ± 0.09. None of the accuracy differences between the consensus and Kruskal-Wallis, "
        "LASSO, stability selection, ACO, MIFS, ANOVA or variance threshold was significant under the corrected resampled "
        "t-test (all p &gt; 0.13; Holm-adjusted p ≥ 0.92). With five folds the benchmark ranks selectors only coarsely, "
        "and small differences in Table 1 should not be read as superiority."),
      P("Rankings also depended on the downstream classifier (Table S4; Spearman correlation of method rankings 0.75 "
        "between kNN and logistic regression and 0.73 between kNN and random forest). The consensus ranked 17th of 33 "
        "with kNN, 3rd with logistic regression (balanced accuracy 0.886) and 1st with random forest (0.932), and had the "
        "highest balanced accuracy averaged over the three classifiers (0.897). Kruskal-Wallis, seventh under kNN, fell "
        "to 26th under random forest, whereas stability selection stayed near the top under all three while remaining "
        "the most stable of the accurate selectors. The kNN ranking in Table 1 is therefore one view, not a verdict."),
      table([["Method", "kNN", "Logistic regression", "Random forest", "Mean", "Ranks (kNN / LR / RF)"],
             ["Consensus", "0.873", "0.886", "0.932", "0.897", "17 / 3 / 1"],
             ["Genetic algorithm", "0.891", "0.893", "0.903", "0.896", "6 / 1 / 5"],
             ["Stability selection", "0.882", "0.874", "0.924", "0.894", "12 / 6 / 2"],
             ["mRMR", "0.883", "0.870", "0.921", "0.891", "11 / 7 / 3"],
             ["Simulated annealing", "0.893", "0.861", "0.901", "0.885", "4 / 9 / 8"],
             ["Boruta", "0.893", "0.858", "0.903", "0.885", "5 / 13 / 6"],
             ["ACO", "0.910", "0.864", "0.878", "0.884", "1 / 8 / 17"],
             ["LASSO", "0.886", "0.860", "0.898", "0.881", "8 / 11 / 10"],
             ["Kruskal-Wallis", "0.889", "0.849", "0.843", "0.860", "7 / 17 / 26"]],
            [0.22, 0.1, 0.16, 0.14, 0.1, 0.28],
            "<i>Table S4. Balanced accuracy of each selector's fold-wise genes under three classifiers (nested 5-fold "
            "cross-validation); ranks among 33 entries (32 selectors and the consensus).</i>"),
      P("In-fold ComBat-style correction raised the pooled balanced accuracy of the consensus genes from 0.873 to 0.904 "
        "but did not remove study structure: a random forest identified the source study in 0.72 of held-out samples "
        "after correction against 0.63 before (baseline 0.35), whereas on simulated additive effects the same procedure "
        "removed it. The study effects in these data are therefore not simple additive shifts, consistent with the "
        "different platforms and sample handling of the four series (Fig. S3). In leave-one-study-out validation, "
        "study-wise centring helped the two studies that lack a class slightly (0.46 to 0.52; 0.86 to 0.88) and hurt the "
        "two complete ones (0.66 to 0.56; 0.85 to 0.83), as expected when class composition differs between studies."),
      P("Only one pair of samples exceeded r = 0.98: a primary and a metastatic sample from GSE15605 (r = 0.994, against "
        "a 99th percentile of 0.93 across all pairs), plausibly paired tissue from one patient. The two fell in different "
        "folds; removing them changed the consensus balanced accuracy from 0.873 to 0.875. Patient identifiers are not "
        "part of the merged matrix, so grouped cross-validation by patient was not possible (Section 5)."),
      P("Repeating the nested pipeline on two further random partitions (15 outer folds in total) left the picture "
        "unchanged. The consensus reached balanced accuracy 0.871 ± 0.062 (95% CI 0.84 to 0.91), again the median of "
        "the 33 entries (median selector 0.870), with stability 0.35 against a median of 0.23. The most accurate "
        "selector changed between partitions, from ACO in the first to forward selection overall (0.900), and no "
        "selector differed significantly from the consensus after Holm correction (smallest adjusted p = 0.08, variance "
        "threshold). PLPPR4 was selected in all 15 outer folds and ranked first in every one; KRT14, KRT16, KRT6A, "
        "S100A7 and SERPINB5 were selected in at least 12."),
      P("With MCP-counter's ten immune and stromal populations, adjustment reduced the three-class balanced accuracy "
        "of the signature from 0.873 to 0.702, and adding keratinocyte and melanocyte content reduced it to 0.635, close "
        "to the 0.665 obtained with our marker scores; between primary and metastatic tumours, 0.767 and 0.737 survived "
        "the two adjustments. MCP-counter scores alone classified at 0.749, against 0.874 for the five-score model, so "
        "epidermal and melanocytic content, not immune or stromal infiltration, carries most of the composition "
        "signal."),
      H2("3.11 External validation and functional context"),
      P("On TCGA-SKCM, the fixed 15-gene signature separated primary from metastatic tumours with AUC 0.796 (kNN) and "
        "0.842 (logistic regression) and balanced accuracy 0.746 for both, without any retraining. Thirteen of the 15 genes "
        "changed in the same direction between primary and metastatic tumours as in the microarray cohort. PLPPR4 was lower "
        "in TCGA metastases than in primaries (difference -0.79 log2 units, Mann-Whitney p = 5 × 10<super>−5</super>), "
        "replicating its drop from primary to metastatic disease, and EMP3 was likewise lower in metastases (p = 7 × "
        "10<super>−10</super>). The keratins, S100A7 and SERPINB5 showed the largest differences (-4.5 to -7.3 log2 units), "
        "consistent with the largely nodal and visceral origin of TCGA metastases. The signature thus transfers across "
        "cohort, platform and measurement technology for the primary-versus-metastatic distinction, while the three-class "
        "task with normal skin remains untested externally."),
      P("Functional enrichment of the top 60 genes in STRING was dominated by epidermal terms: epidermis development (19 "
        "genes, FDR 9 × 10<super>−16</super>), keratinocyte differentiation (FDR 2 × 10<super>−11</super>), cornified "
        "envelope (KEGG, 16 genes, FDR 5 × 10<super>−16</super>) and keratin filament; melanin biosynthesis (TYR, TYRP1; "
        "FDR 0.02) was the only melanocytic term (Table S5). The high-confidence interaction network had 43 edges among 21 "
        "proteins against 2 expected by chance (PPI enrichment p = 2 × 10<super>−41</super>), and 15 of those proteins "
        "formed a single keratin and cornified-envelope module with desmoplakin, loricrin and SPRR1B as its most central "
        "nodes. PLPPR4, EMP3, FABP4, PLOD3 and CDC42EP3 had no high-confidence interaction with any other gene in the list. "
        "The interaction data thus independently reproduce the split between the epidermal module and the "
        "composition-independent candidates seen in the co-expression and composition analyses.")]

s += [H1("4. Discussion"),
      P("The central lesson is a trade-off rather than a winner. The selectors that fit a given split best, the "
        "wrappers and metaheuristics, are the least reproducible, choosing largely different genes in every fold, while "
        "the most reproducible filters rank genes by variance rather than by their relation to disease. A consensus "
        "over all 32 sits between these extremes but does not dominate them: LASSO, Kruskal-Wallis and stability "
        "selection were both more accurate and more stable. For a practitioner with no independent cohort on which to "
        "choose among selectors, the consensus is a defensible default that avoids the worst outcomes; it is not a "
        "route to better markers than a well-chosen single method."),
      P("The consensus refinements did not earn their complexity here. Weighting votes by within-training accuracy and "
        "sharing votes among near-duplicate methods changed balanced accuracy by at most 1.5 points and did not enlarge "
        "the stable core relative to a plain vote count (Table S2). Redundancy control remains a sensible safeguard when "
        "near-identical methods are pooled, as ANOVA, Fisher score and permutation importance were here, but its "
        "benefit should not be assumed."),
      H2("4.1 Biology of the signature"),
      P("Eight of the ten highest-ranked genes belong to the epidermal co-expression module: KRT6A, KRT14, KRT16, ANXA8, "
        "S100A7, SERPINB5, KLF5 and AHNAK. All are reduced in metastases relative to normal skin, with log2 fold changes "
        "between -2.2 and -6.7, and keratin downregulation in melanoma lesions has been reported independently [35]. "
        "Section 3.7 shows how this should be read. These genes correlate with keratinocyte content at r between 0.78 "
        "and 0.96, form a single dense co-expression module, and metastatic samples carry little epidermis. Their "
        "discriminative power is real but largely reports <i>where the sample came from and how much epidermis it "
        "contains</i>, not what the tumour cells are doing."),
      P("SERPINB5 (maspin) illustrates the trap. The literature supports a role in melanoma: Denk et al. [12] found "
        "loss of maspin in melanoma cells relative to normal melanocytes and reduced invasion on re-expression, "
        "Vereecken et al. [13] and Chua et al. [14] reported loss in thicker or vertical-growth-phase lesions, and "
        "Martinoli et al. [36] linked nuclear maspin to prognosis. Yet in bulk tissue SERPINB5 correlates with "
        "keratinocyte score at r = 0.96, the strongest of any signature gene, because maspin is abundantly expressed by "
        "epidermal keratinocytes. Its consensus rank cannot be taken as evidence for the melanoma-cell mechanism; that "
        "requires cell-resolved measurement such as immunohistochemistry."),
      P("S100A7 is in the same position. Psoriasin is expressed in differentiated epidermal layers and has been reported "
        "as absent from melanocytes [15]; its correlation with keratinocyte score here is 0.92. Its selection in all "
        "five folds makes it a reliable <i>classifier</i> of disease state and an uninformative <i>mechanistic</i> "
        "candidate."),
      P("The genes whose signal survives this reading are those outside the epidermal module: PLPPR4, EMP3, FABP4, "
        "PLOD3 and CDC42EP3, with keratinocyte correlations between -0.38 and 0.40. EMP3, an epithelial membrane "
        "protein, correlates instead with melanocytic content and is highest in primary tumours. These, rather than the "
        "top of the consensus ranking taken at face value, are the candidates we would take to immunohistochemistry."),
      P("Composition is not only a confounder. Melanocytic content is part of what distinguishes a melanoma from normal "
        "skin, and immune infiltration is a recognised feature of melanoma biology, so adjusting for them removes biology "
        "as well as sampling artefact. Keratinocyte, fibroblast and endothelial content, by contrast, mostly reflect how "
        "much surrounding skin a biopsy included and where metastases were taken. This is why we treat the epidermal "
        "module as a sampling signal and the genes that survive adjustment as the tumour-associated candidates, and why "
        "the 0.665 balanced accuracy after full adjustment is a lower bound on the tumour-intrinsic signal rather than an "
        "estimate of it."),
      H2("4.2 PLPPR4 and what three-class analyses can see"),
      P("The PLPPR4 result illustrates the methodological argument better than the aggregate numbers do. The gene is "
        "invisible to a comparison of metastases with normal tissue, because its expression rises in primary tumours "
        "and falls back in metastases, so the two extremes look alike. It ranked first in every fold because 23 of the "
        "32 selectors, all trained on three classes, picked up exactly this intermediate-stage information; even a "
        "univariate three-class test detects it. Its moderate correlation with composition (r = 0.40) and isolation "
        "from the epidermal co-expression module make it the strongest tumour-associated candidate in the study."),
      P("A transient marker of primary disease also has a plausible clinical use that a monotonic marker does not: "
        "distinguishing primary tumours from metastatic deposits, the comparison where the signature retained 0.796 "
        "balanced accuracy after full composition adjustment. Because the matrix contains no benign naevi, nothing "
        "here bears on the naevus-versus-melanoma decision, and we make no claim about it."),
      H2("4.3 Why cross-cohort transfer fails, and what it means"),
      P("The honest headline is that performance drops substantially when a whole cohort is held out. Three factors "
        "contribute. Two of the four studies lack a class, so the held-out task differs from the training task; in "
        "Study 1, 36 of 45 primary tumours were called metastatic. Residual platform and study effects survive quantile "
        "normalisation, as the 0.65 study-prediction accuracy shows. And tissue composition differs by study, since the "
        "anatomical sites of metastases, which we could not harmonise, may differ between studies, so a "
        "composition-heavy signature inherits each study's sampling choices."),
      P("We report this rather than suppress it because the pooled-cross-validation numbers that dominate this "
        "literature are systematically optimistic, and because the gap between 0.873 pooled and 0.46-0.86 held-out is "
        "the most useful number in the paper for anyone planning a follow-up study. A signature that classifies well "
        "within a pooled cohort is a hypothesis, not a diagnostic.")]

# ------------------------------------------------------------------ 5-7
s += [H1("5. Limitations and future work"),
      P("<b>No experimental validation.</b> Every result is computational. The signature is a hypothesis for qPCR or "
        "immunohistochemistry on an independent tissue series, not a validated assay."),
      P("<b>Cross-cohort performance is weak.</b> Quantified in Section 3.5 and discussed in 4.3. Any clinical claim "
        "would require a prospective cohort processed on one platform."),
      P("<b>Composition is estimated, not measured.</b> The cell-type scores in Section 3.7 are marker averages, not a "
        "calibrated deconvolution. They are sufficient to show that composition carries much of the signal, but "
        "reference-based methods (CIBERSORTx, xCell) or single-cell data would quantify it more precisely."),
      P("<b>Sampling sites were not harmonised.</b> Normal skin and primary tumours are cutaneous, whereas metastases "
        "can come from lymph nodes, subcutaneous or visceral sites. Site annotations were not harmonised across the four "
        "studies; they should be extracted from the GEO sample records and tested as a covariate. No naevi were "
        "included, so the work says nothing about the naevus-versus-melanoma distinction."),
      P("<b>Consensus weights depend on one classifier.</b> Within-training accuracy is estimated with k-nearest "
        "neighbours. A different base classifier would shift the weights, and we have not measured how much."),
      P("<b>Residual batch effects.</b> Joint quantile normalisation reduced but did not remove study structure; source "
        "study remained predictable at 0.65 accuracy against a 0.35 baseline."),
      P("<b>Patient-level independence could not be verified.</b> Patient identifiers are not part of the merged matrix. A "
        "correlation screen found one probable paired sample, with no effect on the results, but grouped cross-validation "
        "by patient, using identifiers from the GEO sample records, is needed to exclude patient-level leakage."),
      P("<b>Cross-sectional categories.</b> Normal skin, primary and metastatic samples come largely from different "
        "patients. The three categories are not a time course, and expression differences between them are not evidence "
        "of temporal progression."),
      P("<b>Five folds.</b> With five outer folds, confidence intervals are wide and small differences between selectors "
        "cannot be resolved; repeated nested cross-validation would narrow them."),
      P("<b>Microarray only.</b> All four cohorts are microarray. Whether the signature holds in RNA-seq data is "
        "untested."),
      P("<b>Future work.</b> Three steps would turn these candidates into evidence: immunohistochemistry of PLPPR4 and "
        "EMP3 on matched primary and metastatic tissue, which resolves cell of origin directly; external validation on "
        "an independent single-platform cohort with harmonised sampling sites; and transfer to RNA-seq data such as "
        "TCGA-SKCM, which also tests whether the signature survives a change of measurement technology."),
      H1("6. Conclusions"),
      P("Under leakage-free evaluation, 32 feature selectors traded accuracy against stability: the most accurate chose "
        "different genes in every fold, and the most stable were the least accurate. A performance-weighted consensus "
        "reached median accuracy with above-median stability, but LASSO, Kruskal-Wallis and stability selection did "
        "better on both counts, and the consensus refinements made little difference. Three genes, PLPPR4, S100A7 and "
        "KRT14, were nonetheless selected in every fold."),
      P("The most distinctive gene is PLPPR4: chosen by most selectors and first in the consensus in every fold, yet "
        "missed by a metastasis-versus-normal comparison because its expression peaks in primary melanoma and falls back "
        "in metastases. EMP3 also peaks in primary melanoma, though it stays elevated in metastases. Both stand outside "
        "the epidermal co-expression module and are weakly related to tissue composition, which makes them the "
        "candidates with the best case for tumour-associated biology."),
      P("Three results temper the headline. Tissue composition explained most of the three-class signal, so the "
        "highest-ranked keratin genes, S100A7 and SERPINB5 should be read as markers of sample type rather than tumour "
        "mechanism. Holding out whole cohorts reduced balanced accuracy to between 0.46 and 0.86. And selection bias, "
        "though smaller here than the differences between methods, inflated accuracy for three quarters of the "
        "selectors. Pooled cross-validation in merged public datasets is hypothesis generation; the next step is "
        "immunohistochemical validation of PLPPR4 and EMP3 and an independent single-platform cohort."),
      H1("7. Declarations"),
      H2("Data and code availability"),
      P("All expression data are public in the Gene Expression Omnibus under GSE3189, GSE8401, GSE15605 and GSE46517. "
        "Analysis code is provided as core.py (data loading, symbol repair, metrics), fs_methods.py (32 selectors), "
        "fs_methods_reference.py (minepy, boruta_py and Keras implementations), step.py (resumable nested pipeline), "
        "fs_methods_faithful.py and sensitivity.py (full Boruta, exact MIC, dropout network, robustness analysis), "
        "config.py (data and output paths), review_checks.py (selection bias, stability, ablation, and the nested "
        "null-model, published-marker and composition analyses), extra_analyses.py (composition, co-expression "
        "network, ROC), enrichment_ppi.py (GO, KEGG and STRING) and figures.py, with a README and requirements file. "
        + "Code, data and results: https://github.com/majidpilehvar/melanoma-fs-benchmark, archived at Zenodo (DOI to be added)."),
      H2("Author contributions (CRediT)"),
      P("Majid Pilehvar: conceptualisation, methodology, software, formal analysis, investigation, writing (original "
        "draft). Jafar Razmara: supervision, conceptualisation, writing (review and editing). Jaber Karimpour: "
        "supervision, methodology, writing (review and editing). " + note("Confirm roles with supervisors before submission")),
      H2("Funding"),
      P("This research received no specific grant from any funding agency in the public, commercial or not-for-profit "
        "sectors. " + note("Replace if university or grant funding applies")),
      H2("Competing interests"),
      P("The authors declare no competing interests."),
      H2("Ethics statement"),
      P("This study used only publicly available, de-identified gene-expression data from the Gene Expression Omnibus. "
        "No new human or animal subjects were involved, and no ethical approval was required."),
      H2("Acknowledgements"),
      P("This work is based on the first author's MSc thesis in Bioinformatics at the University of Tabriz."),
      H2("Declaration of generative AI and AI-assisted technologies in the writing process"),
      P("During the preparation of this work the authors used Claude (Anthropic) to restructure the analysis code, "
        "audit it for evaluation errors, and draft and edit the manuscript text. After using this tool, the authors "
        "reviewed and edited the content as needed and take full responsibility for the content of the publication. "
        + note("Required by Elsevier policy; adjust the wording to reflect exactly how the tool was used"))]

# ------------------------------------------------------------------ references
R = [
 "Arnold M, Singh D, Laversanne M, et al. Global burden of cutaneous melanoma in 2020 and projections to 2040. <i>JAMA Dermatol</i>. 2022;158(5):495-503.",
 "Ein-Dor L, Kela I, Getz G, Givol D, Domany E. Outcome signature genes in breast cancer: is there a unique set? <i>Bioinformatics</i>. 2005;21(2):171-178.",
 "Michiels S, Koscielny S, Hill C. Prediction of cancer outcome with microarrays: a multiple random validation strategy. <i>Lancet</i>. 2005;365:488-492.",
 "Saeys Y, Inza I, Larrañaga P. A review of feature selection techniques in bioinformatics. <i>Bioinformatics</i>. 2007;23(19):2507-2517. doi:10.1093/bioinformatics/btm344",
 "Guyon I, Weston J, Barnhill S, Vapnik V. Gene selection for cancer classification using support vector machines. <i>Mach Learn</i>. 2002;46:389-422.",
 "Saeys Y, Abeel T, Van de Peer Y. Robust feature selection using ensemble feature selection techniques. In: <i>ECML PKDD 2008</i>. Springer; 2008:313-325. doi:10.1007/978-3-540-87481-2_21",
 "Abeel T, Helleputte T, Van de Peer Y, Dupont P, Saeys Y. Robust biomarker identification for cancer diagnosis with ensemble feature selection methods. <i>Bioinformatics</i>. 2010;26(3):392-398. doi:10.1093/bioinformatics/btp630",
 "Bolón-Canedo V, Alonso-Betanzos A. Ensembles for feature selection: a review and future trends. <i>Inf Fusion</i>. 2019;52:1-12.",
 "Ambroise C, McLachlan GJ. Selection bias in gene extraction on the basis of microarray gene-expression data. <i>Proc Natl Acad Sci USA</i>. 2002;99(10):6562-6566. doi:10.1073/pnas.102102699",
 "Simon R, Radmacher MD, Dobbin K, McShane LM. Pitfalls in the use of DNA microarray data for diagnostic and prognostic classification. <i>J Natl Cancer Inst</i>. 2003;95:14-18.",
 "Lezcano C, Jungbluth AA, Nehal KS, Hollmann TJ, Busam KJ. PRAME expression in melanocytic tumors. <i>Am J Surg Pathol</i>. 2018;42(11):1456-1465. doi:10.1097/PAS.0000000000001134",
 "Denk AE, Bettstetter M, Wild PJ, et al. Loss of maspin expression contributes to a more invasive potential in malignant melanoma. <i>Pigment Cell Res</i>. 2007;20(2). " + note("† pages"),
 "Vereecken P, Reynaert S, Lalmand MC, Zouaoui Boudjeltia K, Heenen M, Van Den Heule B, Petein M. <i>J Int Med Res</i>. 2006;34(1):52-57. PMID 16604823. " + note("† exact title"),
 "Chua R, Setzer S, Govindarajan B, Sexton D, Cohen C, Arbiser JL. Maspin expression, angiogenesis, prognostic parameters, and outcome in malignant melanoma. <i>J Am Acad Dermatol</i>. 2009;60(5):758-766. doi:10.1016/j.jaad.2009.01.018",
 "Moubayed N, Weichenthal M, Harder J, Wandel E, Sticherling M, Gläser R. Psoriasin (S100A7) is significantly up-regulated in human epithelial skin tumours. <i>J Cancer Res Clin Oncol</i>. 2007;133(4):253-261. doi:10.1007/s00432-006-0164-y",
 "Hattinger E, Zwicker S, Ruzicka T, Yuspa SH, Wolf R. Opposing functions of psoriasin (S100A7) and koebnerisin (S100A15) in epithelial carcinogenesis. <i>Curr Opin Pharmacol</i>. 2013;13(4):588-594. doi:10.1016/j.coph.2013.04.007",
 "Barrett T, Wilhite SE, Ledoux P, et al. NCBI GEO: archive for functional genomics data sets, update. <i>Nucleic Acids Res</i>. 2013;41(D1):D991-D995.",
 "Bolstad BM, Irizarry RA, Åstrand M, Speed TP. A comparison of normalization methods for high density oligonucleotide array data based on variance and bias. <i>Bioinformatics</i>. 2003;19(2):185-193.",
 "Talantov D, Mazumder A, Yu JX, Briggs T, Jiang Y, Backus J, Atkins D, Wang Y. Novel genes associated with malignant melanoma but not benign melanocytic lesions. <i>Clin Cancer Res</i>. 2005;11(20):7234-7242. " + note("† pages"),
 "Xu L, Shen SS, Hoshida Y, Subramanian A, Ross K, Brunet JP, Wagner SN, Ramaswamy S, Mesirov JP, Hynes RO. Gene expression changes in an animal melanoma model correlate with aggressiveness of human melanoma metastases. <i>Mol Cancer Res</i>. 2008;6(5):760-769.",
 "Raskin L, Fullen DR, Giordano TJ, et al. Transcriptome profiling identifies HMGA2 as a biomarker of melanoma progression and prognosis. <i>J Invest Dermatol</i>. 2013;133(11):2585-2592. doi:10.1038/jid.2013.197",
 "Kabbarah O, Nogueira C, Feng B, et al. Integrative genome comparison of primary and metastatic melanomas. <i>PLoS One</i>. 2010;5(5):e10770. doi:10.1371/journal.pone.0010770",
 "Tibshirani R. Regression shrinkage and selection via the lasso. <i>J R Stat Soc Series B</i>. 1996;58(1):267-288.",
 "Breiman L. Random forests. <i>Mach Learn</i>. 2001;45:5-32.",
 "Meinshausen N, Bühlmann P. Stability selection. <i>J R Stat Soc Series B</i>. 2010;72(4):417-473.",
 "Kursa MB, Rudnicki WR. Feature selection with the Boruta package. <i>J Stat Softw</i>. 2010;36(11):1-13.",
 "Peng H, Long F, Ding C. Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy. <i>IEEE Trans Pattern Anal Mach Intell</i>. 2005;27(8):1226-1238.",
 "Battiti R. Using mutual information for selecting features in supervised neural net learning. <i>IEEE Trans Neural Netw</i>. 1994;5(4):537-550.",
 "Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. <i>J R Stat Soc Series B</i>. 1995;57(1):289-300.",
 "Newman AM, Steen CB, Liu CL, et al. Determining cell type abundance and expression from bulk tissues with digital cytometry. <i>Nat Biotechnol</i>. 2019;37(7):773-782.",
 "Aran D, Hu Z, Butte AJ. xCell: digitally portraying the tissue cellular heterogeneity landscape. <i>Genome Biol</i>. 2017;18:220.",
 "Szklarczyk D, Kirsch R, Koutrouli M, et al. The STRING database in 2023. <i>Nucleic Acids Res</i>. 2023;51(D1):D638-D646.",
 "Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. <i>J Mach Learn Res</i>. 2011;12:2825-2830.",
 "Reshef DN, Reshef YA, Finucane HK, et al. Detecting novel associations in large data sets. <i>Science</i>. 2011;334(6062):1518-1524.",
 "Nadeau C, Bengio Y. Inference for the generalization error. <i>Mach Learn</i>. 2003;52:239-281.",
 "Holm S. A simple sequentially rejective multiple test procedure. <i>Scand J Stat</i>. 1979;6(2):65-70.",
 "Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray expression data using empirical Bayes methods. <i>Biostatistics</i>. 2007;8(1):118-127.",
 "Becht E, Giraldo NA, Lacroix L, et al. Estimating the population abundance of tissue-infiltrating immune and stromal cell populations using gene expression. <i>Genome Biol</i>. 2016;17:218. doi:10.1186/s13059-016-1070-5",
 "Cancer Genome Atlas Network. Genomic classification of cutaneous melanoma. <i>Cell</i>. 2015;161(7):1681-1696. doi:10.1016/j.cell.2015.05.044",
 "Cerami E, Gao J, Dogrusoz U, et al. The cBio cancer genomics portal: an open platform for exploring multidimensional cancer genomics data. <i>Cancer Discov</i>. 2012;2(5):401-404. doi:10.1158/2159-8290.CD-12-0095",
 "Han W, Xu WH, Wang JX, Hou JM, Zhang HL, Zhao X, Shen G. Identification, validation, and functional annotations of genome-wide profile variation between melanocytic nevus and malignant melanoma. <i>Biomed Res Int</i>. 2020;2020:1840415. doi:10.1155/2020/1840415",
 "Martinoli C, Gandini S, Luise C, et al. Maspin expression and melanoma progression: a matter of sub-cellular localization. <i>Mod Pathol</i>. 2014;27(3):412-419. doi:10.1038/modpathol.2013.157",
 "Christians A, Poisel E, Hartmann C, von Deimling A, Pusch S. Characterization of the epithelial membrane protein 3 interaction network reveals a potential functional link to mitogenic signal transduction regulation. <i>Int J Cancer</i>. 2019;145(2):461-473. doi:10.1002/ijc.32107",
 "Alaminos M, et al. EMP3, a myelin-related gene located in the critical 19q13.3 region, is epigenetically silenced and exhibits features of a candidate tumor suppressor in glioma and neuroblastoma. <i>Cancer Res</i>. 2005. " + note("† authors, volume, pages"),
 "Trimbuch T, Beed P, Vogt J, et al. Synaptic PRG-1 modulates excitatory transmission via lipid phosphate-mediated signaling. <i>Cell</i>. 2009;138:1222-1235. doi:10.1016/j.cell.2009.06.050",
 "Stracke ML, Krutzsch HC, Unsworth EJ, Arestad A, Cioce V, Schiffmann E, Liotta LA. Identification, purification, and partial sequence analysis of autotaxin, a novel motility-stimulating protein. <i>J Biol Chem</i>. 1992;267(4):2524-2529.",
]
s += [H1("References"),
      P("<i>Numbered in order of first citation. Entries marked † still need one detail confirmed against the publisher "
        "record.</i>", ParagraphStyle("rn", parent=body, fontSize=8.6))]
s += [Paragraph(f"[{i}]&nbsp;&nbsp;{r}", ref) for i, r in enumerate(R, 1)]
s += [Spacer(1, 4), P("<b>Datasets.</b> Gene Expression Omnibus series GSE3189 [19], GSE8401 [20], GSE15605 [21] and "
                      "GSE46517 [22], https://www.ncbi.nlm.nih.gov/geo/.", ref)]

# ------------------------------------------------------------------ supplementary figures
s += [PageBreak(), H1("Supplementary figures"),
      figure("FigS1_PCA.png", "<b>Fig. S1.</b> Principal component analysis after joint normalisation, coloured by "
             "disease class (left) and source study (right)."),
      figure("FigS2_coexpression_network.png", "<b>Fig. S2.</b> Co-expression network of the top 60 consensus genes at "
             "|r| ≥ 0.7; node size proportional to betweenness centrality. Genes without a strong partner are omitted "
             "from the drawing and listed in Section 3.8.", 0.86),
      figure("FigS3_batch_correction.png", "<b>Fig. S3.</b> Principal component analysis coloured by source study before "
             "(left) and after (right) a ComBat-style location correction with disease class as a protected covariate.")]

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.0 * cm,
                      bottomMargin=1.9 * cm, title="Benchmarking feature selection for melanoma transcriptomics", author="Majid Pilehvar, Jafar Razmara, Jaber Karimpour",
                      subject="Manuscript draft")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=on_page)])
doc.build(s)
print("written", OUT)
unused=[REPL[i][0][:70] for i,n in USED.items() if n==0]
print("unused replacements:", unused if unused else "none")
