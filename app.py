# app.py
# -*- coding: utf-8 -*-
# Finales UI: Cluster-Karten kompakter, Input-Placeholder grau, Navigation verbreitert.

import io
import os
import math
import collections
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Optional PDF ----------
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except Exception:
    HAS_PYPDF = False

# ---------- Optional ML (TF-IDF) ----------
_HAS_SKLEARN = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    _HAS_SKLEARN = False


# ============ Helpers ============
def do_rerun():
    rr = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rr:
        rr()

def pct(x: float) -> int:
    try:
        return int(round(100 * max(0.0, min(1.0, float(x)))))
    except Exception:
        return 0

def read_uploaded_text(file) -> str:
    if file is None:
        return ""
    name = file.name.lower()
    try:
        if name.endswith((".txt", ".md")):
            return file.read().decode(errors="ignore")[:8000]
        if name.endswith(".pdf") and HAS_PYPDF:
            reader = PdfReader(io.BytesIO(file.read()))
            return "\n".join([(p.extract_text() or "") for p in reader.pages[:12]])[:8000]
    except Exception:
        pass
    return file.name

def parse_skills(v) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return []
    if isinstance(v, str):
        return [t.strip() for t in v.split("|") if t.strip()]
    return [str(v)]


# ============ Page Config ============
st.set_page_config(
    page_title="Ausbildungs-Matcher",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============ Light Theme CSS (BIBB-Style) ============
st.markdown("""
<style>
:root{
  --bg: #FAFCFF;                /* sehr helles Hintergrundweiß mit Blaustich */
  --text: #111827;              /* fast schwarz */
  --muted: #66748B;             /* dezente Sekundärschrift */
  --card: #FFFFFF;
  --card-border: #E6ECF2;
  --soft: #F4F7FB;

  --primary: #5FB548;           /* Grün (BIBB-Anmutung) */
  --primary-d: #3E8A2D;
  --blue: #2C74B3;              /* Blau als Zweitakzent */
}

/* Grundlayout */
.stApp { background: var(--bg); color: var(--text); }
.block-container { padding-top: 0.8rem !important; }

/* Streamlit-Header/Toolbar ausblenden */
header[data-testid="stHeader"] { background: transparent; }
header[data-testid="stHeader"] * { background: transparent; }
div[data-testid="stToolbar"] { display: none !important; }

/* ---------------------------- */
/* TWEAK: Labels und Upload     */
/* ---------------------------- */

/* Labels über Textfeldern und Uploader erzwingen dunkel */
label[data-testid="stWidgetLabel"] p {
    color: var(--text) !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}

/* Upload Dropzone: Weißer Hintergrund, dunkler Rand, dunkler Text */
[data-testid="stFileUploaderDropzone"] {
    background-color: #ffffff !important;
    border: 1px solid #4B5563 !important; /* dunkler Rand */
    color: var(--text) !important;
}
[data-testid="stFileUploaderDropzone"] div, 
[data-testid="stFileUploaderDropzone"] span, 
[data-testid="stFileUploaderDropzone"] small {
    color: var(--text) !important;
}
/* Button innerhalb des Uploaders ("Browse files") */
[data-testid="stFileUploaderDropzone"] button {
    border: 1px solid #ccc;
    color: #000 !important;
    background-color: #f9f9f9;
}

/* ---------------------------- */
/* TWEAK: Inputs (Textarea)     */
/* ---------------------------- */
/* Zwingt das Textfeld weiß mit dunklem Rand (wie gewünscht) */
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #111827 !important;
}
.stTextArea textarea:focus {
    border-color: #2C74B3 !important;
    box-shadow: 0 0 0 1px #2C74B3 !important;
}

/* TWEAK: Placeholder Text in Grau */
.stTextArea textarea::placeholder {
    color: #6B7280 !important;  /* Sichtbares Grau */
    opacity: 1; /* Fix für Firefox */
}

/* ---------------------------- */
/* TWEAK: Buttons               */
/* ---------------------------- */

/* Standard Buttons: Hell mit dunkler Schrift */
.stButton>button {
  border: 1px solid var(--card-border);
  background: #ffffff !important;
  color: var(--text) !important;
  border-radius: 10px;
  padding: 0.45rem 0.8rem;
  font-weight: 500;
}
.stButton>button:hover {
  border-color: #99a;
  background: #f3f4f6 !important;
}

/* TWEAK: Form Submit Button ("Analysieren") */
[data-testid="stFormSubmitButton"] > button {
  border: 2px solid #111827 !important;
  background: #ffffff !important;
  color: #111827 !important;
  font-weight: 700 !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
  background: #f0f0f0 !important;
  border-color: #000 !important;
}

/* Chips/Badges */
.badge, .chip {
  display:inline-block; padding:6px 12px; margin:2px 8px 0 0;
  border:1px solid var(--card-border);
  background: #ffffff;
  color: var(--muted);
  border-radius:999px;
  font-size:12px;
}

/* ---------------------------- */
/* TWEAK: Cluster Dots & Pills  */
/* ---------------------------- */

/* Cluster-Chip (farbig) */
.cchip {
  display:inline-flex; align-items:center; gap:8px;
  padding:6px 12px; margin:2px 10px 0 0;
  border-radius:999px; border:1px solid var(--card-border);
  background: #ffffff;
  font-size:12px; color: var(--text);
  
  /* FIX: Erlaubt Umbruch, damit Text nicht über Box rausgeht */
  white-space: normal !important; 
  max-width: 100%;
  line-height: 1.2;
  text-align: left;
}

/* Punkt: Größer (14px) und nicht stauchbar */
.cdot { 
    width:14px; 
    height:14px; 
    border-radius:50%; 
    display:inline-block; 
    flex-shrink: 0; 
    margin-top: 1px;
}

/* Karten */
.card {
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: 14px;
  padding: 14px;
  box-shadow: 0 2px 10px rgba(17, 24, 39, .06);
}
.card-soft {
  background: var(--soft);
  border: 1px dashed var(--card-border);
  border-radius: 14px; padding: 12px;
}

/* Farbige Cluster-Kacheln */
.tile {
  border-radius:16px; 
  padding: 10px !important; /* TWEAK: Reduziertes Padding (war 14px) für kompaktere Höhe */
  height: 100%;
  color: var(--text);
  border:1px solid var(--card-border);
  box-shadow: 0 2px 8px rgba(17, 24, 39, .04);
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

/* Kleintexte */
.small { font-size: 12px; color: var(--muted); margin-top: 8px; }

/* Progress (Text darunter) */
.progress-footnote { font-size: 12px; color: var(--muted); margin-top: 4px; }

h2, h3 { letter-spacing: .2px; }
</style>
""", unsafe_allow_html=True)


# ============ Data ============
DATA_PATH = os.path.join("data", "ausbildungen_prepared.csv")

REQUIRED_COLS = ["id", "title", "description", "cluster", "skills", "duration_years"]
OPTIONAL_COLS = ["bhg_group", "bibb_url"]

def load_dataset(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"Spalte '{c}' fehlt in {path}")
    df["skills"] = df["skills"].apply(parse_skills)
    for c in OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = ""
    df["title"] = df["title"].astype(str)
    df["cluster"] = df["cluster"].astype(str)
    df["bhg_group"] = df["bhg_group"].astype(str)
    df["bibb_url"] = df["bibb_url"].astype(str)
    return df

DATA = load_dataset(DATA_PATH)
USE_REAL = DATA is not None and _HAS_SKLEARN


# ===== Cluster-Farben (weiche Pastelltöne, stabil je Cluster) =====
SOFT_PALETTE = [
    "#BFE7C7", "#CFE5FA", "#FBE1AE", "#DCD0FB", "#F9C9D4",
    "#CBEDEC", "#F6D6C0", "#D5EDBE", "#E9E4FF", "#FFE9C9",
    "#CDE3F9", "#D8EDC4", "#FFD7BF", "#E2ECFF", "#CFEFE3",
]

def build_cluster_colors(df: pd.DataFrame) -> Dict[str, str]:
    clusters = sorted(df["cluster"].dropna().unique().tolist())
    cmap = {}
    for i, c in enumerate(clusters):
        cmap[c] = SOFT_PALETTE[i % len(SOFT_PALETTE)]
    return cmap

CLUSTER_COLORS = build_cluster_colors(DATA) if DATA is not None else {}

def tile_bg_for(cluster_name: str) -> str:
    col = CLUSTER_COLORS.get(cluster_name, "#E7EEF6")
    # zarter Verlauf + hauchdünner Rahmen
    return (f"background: linear-gradient(180deg, {col} 0%, #FFFFFF 65%);"
            f" border-color: #E9EFF5;")

def cluster_chip_html(name: str) -> str:
    col = CLUSTER_COLORS.get(name, "#E7EEF6")
    return f"<span class='cchip'><span class='cdot' style='background:{col}'></span>{name}</span>"


# ============ Similarity (TF-IDF) ============
def build_vectorizer_and_matrix(df: pd.DataFrame):
    base_texts = (
        df["title"].fillna("").astype(str) + " " +
        df["description"].fillna("").astype(str) + " " +
        df["bhg_group"].fillna("").astype(str) + " " +
        df["cluster"].fillna("").astype(str) + " " +
        df["skills"].apply(lambda lst: ' '.join(lst))
    ).str.strip().tolist()

    vect = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w\w+\b",
        lowercase=True,
    )
    X = vect.fit_transform(base_texts)
    return vect, X

_VECT, _X = (None, None)
if USE_REAL:
    try:
        _VECT, _X = build_vectorizer_and_matrix(DATA)
    except Exception:
        _VECT, _X = (None, None)
        USE_REAL = False

def compute_similarity_tfidf(user_text: str,
                             df: pd.DataFrame,
                             vect: "TfidfVectorizer",
                             X_matrix,
                             top_k: int = 30) -> Tuple[pd.DataFrame, int]:
    txt = (user_text or "").strip()
    if not txt:
        out = df.copy()
        out["similarity"] = 0.0
        out = out.sort_values("title").reset_index(drop=True)
        return out.head(top_k), 0

    q = vect.transform([txt])
    sims = cosine_similarity(q, X_matrix).ravel()
    out = df.copy()
    out["similarity"] = sims
    out = out.sort_values("similarity", ascending=False).reset_index(drop=True)

    center_idx = 0
    if len(out) > 0:
        best_id = out.iloc[0]["id"]
        try:
            center_idx = int(df.index[df["id"] == best_id][0])
        except Exception:
            center_idx = 0

    return out.head(top_k), center_idx


# ============ Session State ============
ss = st.session_state
ss.setdefault("mode", "input")             # "input" | "results" | "detail"
ss.setdefault("last_text", "")
ss.setdefault("results", None)
ss.setdefault("center_idx", None)
ss.setdefault("selected_id", None)
ss.setdefault("sel_cluster", None)
ss.setdefault("sel_group", None)
ss.setdefault("drawer_open", False)


# ============ Components (pure Streamlit) ============
def header():
    st.markdown(f"<h2 style='margin:.2rem 0 .4rem 0;'>Ausbildungs-Matcher</h2>", unsafe_allow_html=True)
    st.caption("Beschreiben oder hochladen → ähnliche Ausbildungsberufe entdecken. "
               "Oder unten über Cluster und Gruppen stöbern.")

def input_panel():
    st.subheader("Eingabe")
    with st.form("input_form", clear_on_submit=False):
        desc = st.text_area(
            "Beschreibung",
            height=180,
            value=ss.get("last_text", ""),
            placeholder="z. B. Netzwerke, Linux, Hardware-Montage, Kundenkontakt …",
        )
        upload = st.file_uploader("Beschreibung als Datei (PDF/TXT/MD)",
                                  type=["pdf", "txt", "md"])
        
        col_a, col_b = st.columns([0.25, 0.75])
        with col_a:
            # Button wird nun via CSS global gestylt
            submitted = st.form_submit_button("Analysieren")
        with col_b:
            st.caption("PDF/Text optional – für kurze Beschreibungen genügt das Textfeld.")

    if submitted:
        text = (desc or "").strip()
        if not text and upload is not None:
            text = read_uploaded_text(upload)
        ss["last_text"] = text

        if not USE_REAL:
            st.error("Ähnlichkeitssuche nicht verfügbar. Bitte `scikit-learn` installieren und Datensatz prüfen.")
            return

        with st.spinner("Analysiere …"):
            res, center_idx = compute_similarity_tfidf(text, DATA, _VECT, _X, top_k=30)
            ss["results"] = res
            ss["center_idx"] = center_idx
            ss["mode"] = "results"
            do_rerun()

def _open_cluster(cname: str):
    st.session_state["sel_cluster"] = cname
    st.session_state["sel_group"] = None

def _open_group(gname: str):
    st.session_state["sel_group"] = gname

def _back_to_groups():
    st.session_state["sel_group"] = None

def _back_to_all():
    st.session_state["sel_cluster"] = None
    st.session_state["sel_group"] = None

def _open_detail(rid: int):
    st.session_state["selected_id"] = rid
    st.session_state["mode"] = "detail"

def _back_home():
    st.session_state["mode"] = "input"
    st.session_state["sel_cluster"] = None
    st.session_state["sel_group"] = None
    st.session_state["selected_id"] = None

def _back_to_results():
    st.session_state["mode"] = "results"
    st.session_state["selected_id"] = None

def cluster_explorer(df: pd.DataFrame, show_back_buttons: bool = False):
    st.subheader("Cluster-Explorer")

    # Breadcrumb
    crumbs = []
    crumbs.append("Alle Cluster" if ss["sel_cluster"] is None else "Alle Cluster")
    if ss["sel_cluster"]:
        crumbs.append(ss["sel_cluster"])
    if ss["sel_group"]:
        crumbs.append(ss["sel_group"])
    st.caption(" / ".join(crumbs))

    # Level 1: Cluster (farbige Kacheln)
    if ss["sel_cluster"] is None:
        counts = df["cluster"].value_counts().sort_index()
        names = counts.index.tolist()
        vals = counts.values.tolist()

        for start in range(0, len(names), 3):
            cols = st.columns(3, gap="large")
            for i in range(3):
                if start + i >= len(names):
                    continue
                cname = names[start + i]
                cnt = int(vals[start + i])
                with cols[i]:
                    # TWEAK: margin für h4 reduziert für kompaktere Höhe
                    st.markdown(
                        f"<div class='tile' style='{tile_bg_for(cname)}'>"
                        f"<h4 style='margin:0 0 2px 0; line-height:1.2;'>{cluster_chip_html(cname)}</h4>"
                        f"<div class='small'>{cnt} Berufe</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.button("Öffnen", key=f"open_cluster_{cname}",
                              on_click=lambda c=cname: _open_cluster(cname=c))
        return

    # Filter auf Cluster
    sub = df[df["cluster"] == ss["sel_cluster"]].copy()

    # Level 2: Gruppen (helle Karten)
    if ss["sel_group"] is None:
        st.write(f"**{cluster_chip_html(ss['sel_cluster'])}** – wähle eine Berufshauptgruppe:",
                 unsafe_allow_html=True)
        groups = sub["bhg_group"].value_counts().sort_index()
        gnames = groups.index.tolist()
        gcnts = groups.values.tolist()

        for start in range(0, len(gnames), 2):
            cols = st.columns(2, gap="large")
            for i in range(2):
                if start + i >= len(gnames):
                    continue
                g = gnames[start + i]
                cnt = int(gcnts[start + i])
                with cols[i]:
                    st.markdown(
                        f"<div class='card'>"
                        f"<div style='display:flex;align-items:center;justify-content:space-between;'>"
                        f"<div><strong>{g}</strong></div>"
                        f"<div class='badge'>{cnt} Berufe</div>"
                        f"</div>"
                        f"</div>", unsafe_allow_html=True
                    )
                    st.button("Öffnen", key=f"open_group_{g}_{start+i}",
                              on_click=lambda gg=g: _open_group(gname=gg))

        if show_back_buttons:
            c1, c2 = st.columns(2)
            with c1:
                st.button("◀ Zurück zu allen Clustern", on_click=_back_to_all)
        return

    # Level 3: Berufe (Liste)
    final = sub[sub["bhg_group"] == ss["sel_group"]].copy()
    st.write(f"**{ss['sel_group']}** – Berufe ({len(final)})")

    for _, row in final.sort_values("title").iterrows():
        title = row["title"]
        dur = row.get("duration_years", np.nan)
        dur_txt = ""
        if pd.notna(dur):
            try:
                yrs = float(dur)
                dur_txt = f"{yrs:.1f}".rstrip("0").rstrip(".").replace(".", ",") + " Jahre"
            except Exception:
                pass
        c1, c2, c3 = st.columns([0.62, 0.20, 0.18])
        with c1:
            st.write(f"**{title}**")
            st.markdown(cluster_chip_html(row.get("cluster","")) + f"<span class='chip'>{row.get('bhg_group','')}</span>", unsafe_allow_html=True)
        with c2:
            st.caption("Ausbildungsdauer")
            st.write(dur_txt or "–")
        with c3:
            st.button("Details",
                      key=f"detail_from_group_{int(row['id'])}",
                      on_click=lambda rid=int(row["id"]): _open_detail(rid))

    if show_back_buttons:
        c1, c2 = st.columns(2)
        with c1:
            st.button("◀ Zurück zu Gruppen", on_click=_back_to_groups)
        with c2:
            st.button("◀ Zurück zu allen Clustern", on_click=_back_to_all)

def hit_tiles(df: pd.DataFrame, results_df: pd.DataFrame | None, n_cols: int = 3):
    st.subheader("Cluster-Übersicht (Treffer)")
    counts = df["cluster"].value_counts().to_dict()
    hit_counts = collections.Counter()
    if results_df is not None and len(results_df):
        for _, row in results_df.iterrows():
            hit_counts[row["cluster"]] += 1

    for idx, cname in enumerate(sorted(counts.keys())):
        if idx % n_cols == 0:
            cols = st.columns(n_cols, gap="large")
        col = cols[idx % n_cols]
        total = counts[cname]
        hits = int(hit_counts.get(cname, 0))
        with col:
            st.markdown(
                f"<div class='tile' style='{tile_bg_for(cname)}'>"
                f"<h4 style='margin:0 0 4px 0; line-height:1.2;'>{cluster_chip_html(cname)}</h4>"
                f"<div class='small'>Gesamt: {total} · Treffer: {hits}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.button("Öffnen", key=f"open_from_hits_{cname}",
                      on_click=lambda c=cname: _open_cluster(cname=c))

def results_list(res_df: pd.DataFrame):
    st.subheader("Top-Treffer")
    if res_df is None or len(res_df) == 0:
        st.caption("Links auf **Neue Suche** klicken und Analysieren.")
        return

    top_vis = min(12, len(res_df))
    for _, row in res_df.head(top_vis).iterrows():
        c1, c2, c3 = st.columns([0.62, 0.20, 0.18])
        with c1:
            st.write(f"**{row['title']}**")
            st.markdown(
                cluster_chip_html(row.get("cluster","")) +
                f"<span class='chip'>{row.get('bhg_group','')}</span>",
                unsafe_allow_html=True
            )
            skills = row["skills"] if isinstance(row.get("skills"), list) else []
            if skills:
                st.caption(" • ".join(skills[:8]))
        with c2:
            st.caption("Ähnlichkeit")
            st.progress(pct(row["similarity"]) / 100.0, text=f"{pct(row['similarity'])}%")
        with c3:
            st.button("Details",
                      key=f"detail_btn_{int(row['id'])}",
                      on_click=lambda rid=int(row['id']): _open_detail(rid))

    if len(res_df) > top_vis:
        with st.expander("Weitere anzeigen"):
            for _, row in res_df.iloc[top_vis:].iterrows():
                p = pct(row["similarity"])
                st.write(f"- **{row['title']}** — *{row.get('cluster','')}* ({p}%)")


def detail_view(df: pd.DataFrame, rid: int, sim_pct: int | None = None):
    row = df[df["id"] == rid].iloc[0]
    ccolor = CLUSTER_COLORS.get(row.get("cluster",""), "#E7EEF6")

    # Farbiger Kopfbereich (zarter Verlauf)
    st.markdown(
        f"<div class='card' style='border-color:{ccolor}; "
        f"<h3 style='margin:0 0 6px 0;'>{row['title']}</h3>"
        f"{cluster_chip_html(row.get('cluster',''))}<span class='chip'>{row.get('bhg_group','')}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    left, right = st.columns([0.62, 0.38], gap="large")

    with left:
        st.subheader("Kurzprofil")
        desc = (row.get("description") or "").strip()
        st.write(desc if desc else "Kein Profiltext vorhanden.")

        st.subheader("Kernkompetenzen")
        skills = row["skills"] if isinstance(row.get("skills"), list) else []
        if skills:
            st.markdown("".join(f"<span class='chip'>{s}</span>" for s in skills), unsafe_allow_html=True)
        else:
            st.caption("Keine Schlagwörter vorhanden.")

    with right:
        if sim_pct is not None:
            st.subheader("Ähnlichkeit")
            st.progress(sim_pct / 100.0, text=f"{sim_pct}%")

        st.subheader("Daten")
        dur_txt = "–"
        y = row.get("duration_years", None)
        try:
            if pd.notna(y):
                dur_txt = (f"{float(y):.1f}".rstrip("0").rstrip(".").replace(".", ",") + " Jahre")
        except Exception:
            pass

        st.write(f"**Ausbildungsdauer:** {dur_txt}")
        st.write(f"**Berufshauptgruppe:** {row.get('bhg_group','') or '–'}")
        st.write(f"**Cluster:** {row.get('cluster','') or '–'}")

        url = (row.get("bibb_url", "") or "").strip()
        if url:
            if hasattr(st, "link_button"):
                st.link_button("Weiterführende Informationen", url)
            else:
                st.markdown(f"[Weiterführende Informationen]({url})")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.button("◀ Zurück zu Ergebnissen", on_click=_back_to_results)
    with c2:
        st.button("◀ Zur Startseite", on_click=_back_home)


# ============ Layout / Routing ============
if DATA is None:
    st.error("Datensatz nicht gefunden. Bitte Preprocessing ausführen und Datei unter `data/ausbildungen_prepared.csv` bereitstellen.")
else:
    header()

    if ss["mode"] == "input":
        left, right = st.columns([0.52, 0.48], gap="large")
        with left:
            input_panel()
        with right:
            cluster_explorer(DATA, show_back_buttons=True)

    elif ss["mode"] == "results":
        if ss.get("drawer_open", False):
            # Drawer offen
            colL, colC, colR = st.columns([0.28, 0.42, 0.30], gap="large")
        else:
            # Drawer geschlossen -> Navigation (links) breiter gemacht (0.15 statt 0.10)
            colL, colC, colR = st.columns([0.15, 0.50, 0.35], gap="large")

        with colL:
            st.subheader("Navigation")
            if not ss.get("drawer_open", False):
                st.button("Neue Suche", key="open_drawer", use_container_width=True,
                          on_click=lambda: st.session_state.update({"drawer_open": True}))
                st.button("◀ Zur Startseite", use_container_width=True, on_click=_back_home)
            else:
                with st.form("drawer_form", clear_on_submit=False):
                    desc_sb = st.text_area(
                        "Beschreibung",
                        height=140,
                        value=ss.get("last_text", ""),
                        placeholder="z. B. Netzwerke, Linux, Hardware-Montage, Kundenkontakt …",
                    )
                    upload_sb = st.file_uploader(
                        "Beschreibung als Datei (PDF/TXT/MD)",
                        type=["pdf", "txt", "md"], key="upload_sb",
                    )
                    col_a, col_b = st.columns([0.55, 0.45])
                    with col_a:
                        # Auch hier Button-Styling via CSS
                        submitted_sb = st.form_submit_button("Analysieren")
                    with col_b:
                        st.caption("Beschreibung oder Datei genügt.")

                c1, c2 = st.columns(2)
                with c1:
                    st.button("◀ Schließen", use_container_width=True,
                              on_click=lambda: st.session_state.update({"drawer_open": False}))
                with c2:
                    st.button("Zur Startseite", use_container_width=True, on_click=_back_home)

                if submitted_sb:
                    text = (desc_sb or "").strip()
                    if not text and upload_sb is not None:
                        text = read_uploaded_text(upload_sb)
                    ss["last_text"] = text

                    if not USE_REAL:
                        st.error("Ähnlichkeitssuche nicht verfügbar. Bitte scikit-learn installieren.")
                    else:
                        with st.spinner("Analysiere …"):
                            res, center_idx = compute_similarity_tfidf(text, DATA, _VECT, _X, top_k=30)
                            ss["results"], ss["center_idx"] = res, center_idx
                            st.session_state["drawer_open"] = False
                            do_rerun()

        with colC:
            results_list(ss.get("results"))

        with colR:
            # TWEAK: "Cluster-Explorer" hier entfernt, nur noch Hit-Tiles
            hit_tiles(DATA, ss.get("results"), n_cols=1)

    elif ss["mode"] == "detail":
        rid = ss.get("selected_id")
        if rid is None:
            st.info("Kein Beruf ausgewählt.")
        else:
            sim_pct = None
            if ss.get("results") is not None:
                try:
                    sim_val = float(ss["results"][ss["results"]["id"] == rid]["similarity"].values[0])
                    sim_pct = pct(sim_val)
                except Exception:
                    sim_pct = None
            detail_view(DATA, rid, sim_pct)