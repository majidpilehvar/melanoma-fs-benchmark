"""
MCP-counter (Becht et al., Genome Biol 2016) re-implemented from its published
marker table (github.com/ebecht/MCPcounter, Signatures/genes.txt): each
population's score is the mean log2 expression of its marker genes, exactly as
MCPcounter.estimate() computes it. Used to repeat the nested composition
adjustment of Table 5 with an established marker set.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, RESULTS, CKPT
from core import load_data, evaluate_predictions, RANDOM_STATE
from extra_analyses import residualise_in_fold, cell_scores
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

MCP = {
 "T cells": ["CD28", "CD3D", "CD3G", "CD5", "CD6", "CHRM3-AS2", "CTLA4", "FLT3LG", "ICOS", "MAL", "MGC40069", "PBX4", "SIRPG", "THEMIS", "TNFRSF25", "TRAT1"],
 "CD8 T cells": ["CD8B"],
 "Cytotoxic lymphocytes": ["CD8A", "EOMES", "FGFBP2", "GNLY", "KLRC3", "KLRC4", "KLRD1"],
 "B lineage": ["BANK1", "CD19", "CD22", "CD79A", "CR2", "FCRL2", "IGKC", "MS4A1", "PAX5"],
 "NK cells": ["CD160", "KIR2DL1", "KIR2DL3", "KIR2DL4", "KIR3DL1", "KIR3DS1", "NCR1", "PTGDR", "SH2D1B"],
 "Monocytic lineage": ["ADAP2", "CSF1R", "FPR3", "KYNU", "PLA2G7", "RASSF4", "TFEC"],
 "Myeloid dendritic cells": ["CD1A", "CD1B", "CD1E", "CLEC10A", "CLIC2", "WFDC21P"],
 "Neutrophils": ["CA4", "CEACAM3", "CXCR1", "CXCR2", "CYP4F3", "FCGR3B", "HAL", "KCNJ15", "MEGF9", "SLC25A37", "STEAP4", "TECPR2", "TLE3", "TNFRSF10C", "VNN3"],
 "Endothelial cells": ["ACVRL1", "APLN", "BCL6B", "BMP6", "BMX", "CDH5", "CLEC14A", "DIPK2B", "EDN1", "ADGRL4", "EMCN", "ESAM", "ESM1", "FAM124B", "HECW2", "HHIP", "KDR", "MMRN1", "MMRN2", "MYCT1", "PALMD", "PEAR1", "PGF", "PLXNA2", "PTPRB", "ROBO4", "SDPR", "SHANK3", "SHE", "TEK", "TIE1", "VEPH1", "VWF"],
 "Fibroblasts": ["COL1A1", "COL3A1", "COL6A1", "COL6A2", "DCN", "GREM1", "PAMR1", "TAGLN"],
}
# older symbols used on GPL96/GPL570 annotations
ALIASES = {"DIPK2B": ["CXorf36"], "ADGRL4": ["ELTD1"], "SDPR": ["CAVIN2"], "CXCR1": ["IL8RA"], "CXCR2": ["IL8RB"],
           "WFDC21P": ["LOC645638"]}


def mcp_scores(X):
    out, used = {}, {}
    for pop, genes in MCP.items():
        g = []
        for s in genes:
            for cand in [s] + ALIASES.get(s, []):
                if cand in X.columns:
                    g.append(cand); break
        used[pop] = g
        out[pop] = X[g].mean(axis=1) if g else pd.Series(np.nan, index=X.index)
    return pd.DataFrame(out), used


def main():
    X, y, _, _ = load_data(DATA, verbose=False)
    Xv, yv = X.values, y.values; genes = list(X.columns)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE).split(Xv, yv))
    S, used = mcp_scores(X)
    S = S.loc[:, S.notna().all()]
    CK = [json.load(open(f"{CKPT}/fold{f}.json")) for f in range(5)]
    sig_union = set().union(*[set(c["signature"]) for c in CK])
    overlap = {p: sorted(set(g) & sig_union) for p, g in used.items() if set(g) & sig_union}
    excl = set(np.load(f"{CKPT}/top_genes.npy", allow_pickle=True)) | sig_union
    own, _ = cell_scores(X, exclude=excl)
    Zm = ((S - S.mean()) / S.std()).values
    Zk = ((own[["keratinocyte", "melanocyte"]] - own[["keratinocyte", "melanocyte"]].mean()) / own[["keratinocyte", "melanocyte"]].std()).values
    variants = {"mcp_counter_10": Zm, "mcp_counter_plus_keratinocyte_melanocyte": np.hstack([Zm, Zk])}
    res = {"markers_found": {p: len(g) for p, g in used.items()}, "markers_total": {p: len(g) for p, g in MCP.items()},
           "marker_overlap_with_signatures": overlap,
           "scores_by_class": S.groupby(y.values).mean().round(3).to_dict()}
    for task, mask, labels in [("three_class", np.ones(len(yv), bool), (1, 2, 3)), ("primary_vs_metastatic", np.isin(yv, [2, 3]), (2, 3))]:
        out = {}
        for vname, C in variants.items():
            r_sig, r_adj, r_only = [], [], []
            for f, (tr, te) in enumerate(folds):
                tr, te = tr[mask[tr]], te[mask[te]]
                idx = [genes.index(g) for g in CK[f]["signature"]]
                G = Xv[:, idx]
                ev = lambda A, B: evaluate_predictions(yv[te], KNeighborsClassifier(5).fit(A, yv[tr]).predict(B), labels=labels)["balanced_accuracy"]
                r_sig.append(ev(G[tr], G[te]))
                Gtr, Gte = residualise_in_fold(G, C, tr, te); r_adj.append(ev(Gtr, Gte))
                r_only.append(ev(C[tr], C[te]))
            out[vname] = {"signature": round(float(np.mean(r_sig)), 3), "signature_adjusted": round(float(np.mean(r_adj)), 3),
                          "scores_alone": round(float(np.mean(r_only)), 3)}
        res[task] = out
    os.makedirs(f"{RESULTS}/revision", exist_ok=True)
    json.dump(res, open(f"{RESULTS}/revision/mcpcounter.json", "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
