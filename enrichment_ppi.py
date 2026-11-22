"""
GO / KEGG enrichment and STRING protein-protein interaction analysis of the
top-60 consensus genes. Needs internet access (Enrichr libraries, STRING API):

    pip install gseapy requests networkx
    python3 enrichment_ppi.py

Outputs (results/):
    table10_enrichment_GO_BP.csv     GO Biological Process, FDR-ranked
    table10_enrichment_KEGG.csv      KEGG pathways, FDR-ranked
    table11_string_network.csv       degree and betweenness per protein
    string_edges.tsv                 edges with STRING combined score
"""
import os, io, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import DATA, RESULTS as OUT


def load_genes():
    from core import load_data
    X, _, _, _ = load_data(DATA, verbose=False)
    top60 = list(np.load(f"{OUT}/ckpt/top_genes.npy", allow_pickle=True))
    return top60, list(X.columns)


def enrichment(top60, background):
    import gseapy as gp
    libs = {"GO_BP": "GO_Biological_Process_2023", "KEGG": "KEGG_2021_Human"}
    for short, lib in libs.items():
        # background = every gene on the merged array, so enrichment is judged
        # against what could have been selected, not the whole genome
        res = gp.enrichr(gene_list=top60, gene_sets=lib, background=background,
                         organism="human", outdir=None, cutoff=1.0)
        df = res.results.sort_values("Adjusted P-value")
        df.to_csv(f"{OUT}/table10_enrichment_{short}.csv", index=False)
        sig = df[df["Adjusted P-value"] < 0.05]
        print(f"{short}: {len(sig)} terms at FDR < 0.05")
        print(sig[["Term", "Overlap", "Adjusted P-value", "Genes"]].head(10).to_string(index=False))


def string_network(top60, required_score=700):
    import requests
    import networkx as nx
    url = "https://string-db.org/api/tsv/network"
    r = requests.post(url, data={"identifiers": "\r".join(top60), "species": 9606,
                                 "required_score": required_score,
                                 "caller_identity": "melanoma_consensus_fs"}, timeout=60)
    r.raise_for_status()
    e = pd.read_csv(io.StringIO(r.text), sep="\t")
    e = e[["preferredName_A", "preferredName_B", "score"]].drop_duplicates()
    e.to_csv(f"{OUT}/string_edges.tsv", sep="\t", index=False)
    G = nx.from_pandas_edgelist(e, "preferredName_A", "preferredName_B", ["score"])
    tab = pd.DataFrame({"degree": dict(G.degree()),
                        "betweenness": nx.betweenness_centrality(G)}
                       ).sort_values("betweenness", ascending=False)
    tab.to_csv(f"{OUT}/table11_string_network.csv")
    print(f"STRING (score >= {required_score}): {G.number_of_nodes()} proteins, "
          f"{G.number_of_edges()} interactions; "
          f"{len(set(top60) - set(G.nodes))} of 60 genes without a high-confidence partner")
    print(tab.head(10).round(4).to_string())


if __name__ == "__main__":
    top60, background = load_genes()
    enrichment(top60, background)
    string_network(top60)
