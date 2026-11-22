"""
Look for patient identifiers in the GEO sample records of the four series, to
check whether any patient contributes samples to more than one fold.

    pip install GEOparse pandas
    python3 geo_patient_check.py
Needs internet. Writes results/revision/geo_sample_characteristics.csv and
prints any characteristic field that looks like a patient or donor identifier.
"""
import os, sys, re
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, RESULTS
import GEOparse

samples = set(pd.read_csv(DATA, index_col=0, nrows=0).columns)
rows = []
for gse in ["GSE3189", "GSE8401", "GSE15605", "GSE46517"]:
    g = GEOparse.get_GEO(geo=gse, destdir=os.path.join(RESULTS, "geo_cache"), silent=True)
    for gsm, s in g.gsms.items():
        if gsm not in samples:
            continue
        rec = {"series": gse, "gsm": gsm, "title": s.metadata.get("title", [""])[0],
               "source": s.metadata.get("source_name_ch1", [""])[0]}
        for i, c in enumerate(s.metadata.get("characteristics_ch1", [])):
            k, _, v = c.partition(":")
            rec[k.strip().lower() or f"char{i}"] = v.strip()
        rows.append(rec)
df = pd.DataFrame(rows)
os.makedirs(f"{RESULTS}/revision", exist_ok=True)
df.to_csv(f"{RESULTS}/revision/geo_sample_characteristics.csv", index=False)
print(f"{len(df)} of {len(samples)} samples annotated")
for col in df.columns:
    if re.search(r"patient|individual|subject|donor|case|person|id\b", col):
        dup = df[col].dropna().duplicated(keep=False).sum()
        print(f"possible identifier field '{col}': {df[col].nunique()} values, {dup} samples share a value")
print("Also inspect 'title' and 'source': some series encode the patient in the sample title.")
