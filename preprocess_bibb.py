# -*- coding: utf-8 -*-
"""
Preprocess BIBB JSON (already clustered) into a compact CSV for the app.

Input (expected): ./data/bibb_berufe_thresh_0_8.json
  Required keys per row:
    - beruf (title), gruppe (Berufshauptgruppe), cluster (final cluster label),
      dauer_monate, bibb_url, bibb_fliesstext (optional),
      berufliche_taetigkeitsfelder (list/str), profil_berufliche_handlungsfaehigkeit (list/str)

Output: ./data/ausbildungen_prepared.csv
  Columns:
    id, title, description, bhg_group, cluster, duration_years,
    bibb_url, skills, x, y
  (skills is pipe-separated string for Streamlit app convenience)
"""
from __future__ import annotations
import re, json, math, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# Optional ML stack for 2D projection
_HAS_SKLEARN = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD, PCA
except Exception:
    _HAS_SKLEARN = False

DATA_DIR = Path(".") / "data"
IN_JSON  = DATA_DIR / "bibb_berufe_thresh_0_8.json"
OUT_CSV  = DATA_DIR / "ausbildungen_prepared.csv"

def _as_list(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return []
    if isinstance(x, list): return x
    parts = re.split(r"[•●\-–—;\n\r\t]+", str(x))
    return [p.strip(" ·•\t\r\n-–—") for p in parts if p.strip()]

def _mk_desc(row: dict) -> str:
    # Prefer rich text if present
    txt = (row.get("bibb_fliesstext") or "").strip()
    if txt: return txt
    tfs = ", ".join(_as_list(row.get("berufliche_taetigkeitsfelder"))[:8])
    prof = ", ".join(_as_list(row.get("profil_berufliche_handlungsfaehigkeit"))[:8])
    glueA = f"Tätigkeitsfelder: {tfs}" if tfs else ""
    glueB = f"; Profil: {prof}" if prof else ""
    title = (row.get("beruf") or row.get("title") or "").strip()
    return (title + " " + glueA + glueB).strip()

def _mk_skills(row: dict, maxn=12) -> list[str]:
    words = []
    words += _as_list(row.get("berufliche_taetigkeitsfelder"))
    words += _as_list(row.get("profil_berufliche_handlungsfaehigkeit"))
    cleaned, seen = [], set()
    for w in words:
        # keep letters, digits, underscore, space, / + & - (with German umlauts)
        w2 = re.sub(r"[^0-9A-Za-zÄÖÜäöüß_ /+&\-]+", "", str(w)).strip()
        if 2 <= len(w2) <= 48:
            lw = w2.lower()
            if lw not in seen:
                cleaned.append(w2); seen.add(lw)
    return cleaned[:maxn]

def _vectorize_and_project(texts: list[str], max_features=25000, svd_dim=64, map_scale_x=220, map_scale_y=180):
    # ensure non-empty texts
    safe = []
    for i, t in enumerate(texts):
        t = (t or "").strip()
        if not t: t = f"doc{i}"
        safe.append(t)

    if not _HAS_SKLEARN:
        # tidy grid fallback
        n = len(safe)
        xs = np.linspace(-1, 1, max(2, int(math.ceil(math.sqrt(n)))))
        xv, yv = np.meshgrid(xs, xs)
        pts = np.stack([xv.ravel()[:n], yv.ravel()[:n]], axis=1)
    else:
        vect = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), token_pattern=r"(?u)\b\w\w+\b")
        X = vect.fit_transform(safe)
        if X.shape[0] < 3 or X.shape[1] < 3:
            pts = np.stack([np.linspace(-1,1,len(safe)), np.zeros(len(safe))], axis=1)
        else:
            k = max(2, min(svd_dim, X.shape[1]-1, X.shape[0]-1))
            svd = TruncatedSVD(n_components=k, random_state=42)
            Xr = svd.fit_transform(X)
            pca = PCA(n_components=2, random_state=42)
            pts = pca.fit_transform(Xr)

    pts = (pts - pts.mean(0)) / (pts.std(0) + 1e-9)
    pts[:,0] *= map_scale_x; pts[:,1] *= map_scale_y
    return pts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_features", type=int, default=25000)
    parser.add_argument("--svd", type=int, default=64)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not IN_JSON.exists():
        raise SystemExit(f"Input nicht gefunden: {IN_JSON}")

    raw = pd.read_json(IN_JSON)
    if args.verbose:
        print(f"INFO | Quelle geladen: {len(raw)}")

    # normalize
    rows = []
    for _, r in raw.iterrows():
        d = r.to_dict()
        title = (d.get("beruf") or d.get("title") or "").strip()
        gruppe = (d.get("gruppe") or d.get("bhg_group") or "").strip()
        cluster = (d.get("cluster") or "").strip()
        dauer_m = pd.to_numeric(d.get("dauer_monate"), errors="coerce")
        url = (d.get("bibb_url") or d.get("url") or "").strip()

        desc = _mk_desc(d)
        skills = _mk_skills(d)
        rows.append({
            "title": title,
            "bhg_group": gruppe,
            "cluster": cluster,
            "duration_years": (float(dauer_m)/12.0 if pd.notna(dauer_m) else np.nan),
            "bibb_url": url,
            "skills": "|".join(skills),
            "description": desc,
        })

    df = pd.DataFrame(rows).dropna(subset=["title"]).reset_index(drop=True)
    df.insert(0, "id", np.arange(len(df), dtype=int))

    # projection text
    texts = (df["description"].fillna("").astype(str) + " " +
             df["skills"].fillna("").astype(str) + " " +
             df["title"].fillna("").astype(str) + " " +
             df["bhg_group"].fillna("").astype(str) + " " +
             df["cluster"].fillna("").astype(str))
    pts = _vectorize_and_project(texts.tolist(), args.max_features, args.svd)
    df["x"] = pts[:,0]; df["y"] = pts[:,1]

    OUT_CSV.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(OUT_CSV, index=False)
    if args.verbose:
        print(f"INFO | Geschrieben: {OUT_CSV.resolve()}")
        print("INFO | Spalten:", ", ".join(df.columns))

if __name__ == "__main__":
    main()