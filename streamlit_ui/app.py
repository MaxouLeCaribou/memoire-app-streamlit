# app.py — Recherche simple (TF‑IDF) avec 2 corpus: Mémoires / Attestations
import os, sys, io, subprocess, time, re, json
import streamlit as st
import chromadb
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# --- Shim sqlite for Chroma on Streamlit Cloud ---
import sys
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# -------------------------------------------------

import chromadb


# --- FOND ÉCRAN via base64 (fiable) ---
import base64, mimetypes
from pathlib import Path

def set_background(img_path: str):
    p = Path(img_path)
    if not p.is_file():
        import streamlit as st
        st.warning(f"Image introuvable : {p.resolve()}")
        return
    mime = mimetypes.guess_type(p.name)[0] or "image/jpg"
    data = base64.b64encode(p.read_bytes()).decode()

    import streamlit as st
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
      background-image: url("data:{mime};base64,{data}");
      background-repeat: no-repeat;
      background-position: center center;
      background-attachment: fixed;
      background-size: cover;
      position: relative;
    }}

    [data-testid="stAppViewContainer"]::before {{
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0,0,0,0.8); /* assombrissement léger */
      z-index: 0;
    }}

    /* contenu au-dessus du voile */
    [data-testid="stAppViewContainer"] * {{
      position: relative;
      z-index: 1;
    }}
    </style>
    """, unsafe_allow_html=True)


# 👉 Appelle ta fonction avec ton fichier
set_background("wp2932668-noir-wallpaper.jpg")




# ----- Patch import vers le projet -----
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Outils internes (page guess / ouverture PDF)
from app_logic.rc_tools import guess_page_for_snippet
from app_logic.rc_tools import open_pdf_external as open_pdf_in_browser

st.set_page_config(page_title="Recherche PDF", page_icon="🔎", layout="wide")

# ----- Page / CSS -----
c1, c2, c3 = st.columns([1,1,6])
with c1: st.page_link("app.py", label=" Recherche", icon="🔎")
with c2: st.page_link("pages/02_Mode_RC.py", label=" Auto Mém", icon="📁")
with c3: st.write("")
st.divider()

st.markdown("""
<style>
[data-testid="stAppViewContainer"] .main .block-container{
  max-width: 1600px; padding-left:1rem; padding-right:1rem;
}
div[data-testid="stExpander"] > div{ width:100%; }
.small { color:#666; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

# =========================
#  Dossiers & Config (persistés)
# =========================
CONFIG_DIR         = os.path.join(ROOT, "config")
APP_CONFIG_FILE    = os.path.join(CONFIG_DIR, "app_config.json")
DEFAULT_INDEX_MEM  = os.path.join(ROOT, "chroma_index_mem")
DEFAULT_INDEX_ATT  = os.path.join(ROOT, "chroma_index_attest")

os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(DEFAULT_INDEX_MEM, exist_ok=True)
os.makedirs(DEFAULT_INDEX_ATT, exist_ok=True)

def _load_app_cfg():
    try:
        if os.path.isfile(APP_CONFIG_FILE):
            with open(APP_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}

def _save_app_cfg(updates: dict):
    cfg = _load_app_cfg()
    cfg.update({k: v for k, v in updates.items() if v is not None})
    with open(APP_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def get_docs_path() -> str:
    # Dossier "Mémoires"
    p = st.session_state.get("docs_path")
    if p is not None:
        return p
    p = (_load_app_cfg().get("docs_path") or "").strip()
    st.session_state.docs_path = p
    return p

def set_docs_path(value: str):
    v = (value or "").strip()
    st.session_state.docs_path = v
    _save_app_cfg({"docs_path": v})

def get_attest_path() -> str:
    p = st.session_state.get("attest_path")
    if p is not None:
        return p
    p = (_load_app_cfg().get("attest_path") or "").strip()
    st.session_state.attest_path = p
    return p

def set_attest_path(value: str):
    v = (value or "").strip()
    st.session_state.attest_path = v
    _save_app_cfg({"attest_path": v})

def get_last_corpus() -> str:
    c = st.session_state.get("active_corpus")
    if c is not None:
        return c
    c = (_load_app_cfg().get("active_corpus") or "Mémoires").strip()
    st.session_state.active_corpus = c
    return c

def set_last_corpus(value: str):
    v = value if value in ("Mémoires", "Attestations") else "Mémoires"
    st.session_state.active_corpus = v
    _save_app_cfg({"active_corpus": v})

# =========================
#  Chroma (sans embeddings) + TF‑IDF
# =========================
def get_collection(persist_dir: str):
    """
    Aucune embedding_function : Chroma sert de stockage (documents + métadonnées).
    """
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name="memoire_chunks")

@st.cache_resource(show_spinner=False)
@st.cache_resource(show_spinner=False)
def _build_tfidf_index(persist_dir: str):
    """
    Charge tous les chunks depuis Chroma et construit un index TF‑IDF HYBRIDE :
    - TF‑IDF "mots" (ngram 1‑2)
    - TF‑IDF "caractères" (ngram 3‑5, analyzer='char_wb')
    On concatène les matrices pour une recherche robuste sur les noms propres.
    """
    coll = get_collection(persist_dir)
    batch = coll.get(include=["documents", "metadatas"])
    docs  = batch.get("documents", []) or []
    metas = batch.get("metadatas", []) or []

    docs  = [d if isinstance(d, str)  else ""  for d in docs]
    metas = [m if isinstance(m, dict) else {} for m in metas]

    # TF‑IDF "mots"
    vect_word = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents="unicode",
        min_df=1,
        max_df=0.9,
    )
    Xw = vect_word.fit_transform(docs)

    # TF‑IDF "caractères"
    vect_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        strip_accents="unicode",
        min_df=1,
        max_df=1.0,
    )
    Xc = vect_char.fit_transform(docs)

    # Concatène
    from scipy.sparse import hstack
    X = hstack([Xw, Xc]).tocsr()

    return {"vect_word": vect_word, "vect_char": vect_char, "X": X, "docs": docs, "metas": metas}


def tfidf_retrieve(persist_dir: str, query: str, topk: int = 5):
    idx = _build_tfidf_index(persist_dir)
    vect_word, vect_char, X, docs, metas = (
        idx["vect_word"], idx["vect_char"], idx["X"], idx["docs"], idx["metas"]
    )
    if not docs:
        return [], [], []

    # Variantes simples pour noms propres (ex: "leblanc" -> "le blanc")
    q = (query or "").strip()
    variants = {q}
    if " " not in q and len(q) >= 6:
        # insère un espace probable (ex: leblanc -> le blanc)
        variants.add(re.sub(r"([a-z]{2,})([A-Z]|blanc)", r"\1 \2", q, flags=re.I))
        # remplace tirets / points
        variants.add(q.replace("-", " "))
        variants.add(q.replace(".", " "))

    # Construit la requête hybride (somme des similitudes)
    import numpy as np
    sims_total = np.zeros(len(docs), dtype=float)
    for v in variants:
        qv_w = vect_word.transform([v])
        qv_c = vect_char.transform([v])
        from sklearn.metrics.pairwise import cosine_similarity
        sw = cosine_similarity(qv_w, X[:, :qv_w.shape[1]])[0] if qv_w.shape[1] else 0
        # Décale pour la partie char_wb
        start = qv_w.shape[1]
        sc = cosine_similarity(qv_c, X[:, start:start + qv_c.shape[1]])[0] if qv_c.shape[1] else 0
        sims = sw + sc
        sims_total = np.maximum(sims_total, sims)  # garde le meilleur variant

    # Top indices + tri
    top_idx = np.argsort(-sims_total)[:max(topk * 8, topk)]
    cand = [(int(i), float(sims_total[i])) for i in top_idx]
    cand.sort(key=lambda t: t[1], reverse=True)

    out_docs, out_metas, out_dists = [], [], []
    for i, sc in cand[:topk]:
        out_docs.append(docs[i])
        out_metas.append(metas[i])
        out_dists.append(1.0 - sc)  # distance simulée
    return out_docs, out_metas, out_dists


def filter_unique_by_pdf(docs, metas, dists, max_k: int):
    """
    Déduplication stricte par (source_normalisée, page_1based).
    Une même page d'un même PDF ne peut apparaître qu'une fois.
    """
    seen = set()
    f_docs, f_metas, f_dists = [], [], []
    for d, m, dist in zip(docs, metas, dists):
        m = m or {}
        src = (m.get("source") or "").strip()
        page = m.get("page", 1)
        try:
            page = int(page) if (isinstance(page, int) and page >= 1) else 1
        except Exception:
            page = 1
        key = (os.path.normpath(src).lower(), page) if src else ("__no_source__", page)
        if key in seen:
            continue
        seen.add(key)
        f_docs.append(d); f_metas.append(m); f_dists.append(dist)
        if len(f_docs) >= max_k:
            break
    return f_docs, f_metas, f_dists

# =========================
#  Sélection corpus actif
# =========================
def get_active_base_dir() -> str:
    return (get_docs_path() if st.session_state.active_corpus == "Mémoires" else get_attest_path()) or ""

def get_active_index_dir() -> str:
    return DEFAULT_INDEX_MEM if st.session_state.active_corpus == "Mémoires" else DEFAULT_INDEX_ATT

# =========================
#  PDF utils
# =========================
def _resolve_pdf_path(src: str) -> str | None:
    """
    Résout un chemin absolu pour src :
    - direct (abs/rel),
    - ROOT/src,
    - recherche du basename dans le dossier du corpus ACTIF (Mémoires/Attestations).
    """
    if not src:
        return None
    try:
        if os.path.isabs(src) and os.path.isfile(src):
            return os.path.normpath(src)
        if os.path.isfile(src):
            return os.path.abspath(src)
        p2 = os.path.normpath(os.path.join(ROOT, src))
        if os.path.isfile(p2):
            return p2
        base = os.path.basename(src)
        d = get_active_base_dir()
        if d and os.path.isdir(d) and base:
            for r, _, files in os.walk(d):
                if base in files:
                    return os.path.normpath(os.path.join(r, base))
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def _render_pdf_page_png(abs_path: str, page_index0: int, zoom: float = 2.2) -> bytes:
    doc = fitz.open(abs_path)
    try:
        page_count = doc.page_count
        if page_index0 < 0 or page_index0 >= page_count:
            page_index0 = max(0, min(page_index0, page_count-1))
        page = doc.load_page(page_index0)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()

def _find_first_snippet_for_source(src_path: str) -> str:
    """Premier chunk pour ce PDF dans le DERNIER résultat (utile pour deviner la page)."""
    docs  = st.session_state.get("last_docs")  or []
    metas = st.session_state.get("last_metas") or []
    norm  = os.path.normpath(str(src_path)).lower()
    for d, m in zip(docs, metas):
        m = m or {}
        src = os.path.normpath(str(m.get("source",""))).lower()
        if src == norm and d:
            return d
    return ""

def _read_file_bytes(abs_path: str) -> bytes:
    with open(abs_path, "rb") as f:
        return f.read()

def build_pdf_items_from_metas(metas):
    """Construction des cartes PDF à partir des métadonnées (dédup (source,page))."""
    seen = set()
    items = []
    for m in metas or []:
        m = m or {}
        src = (m.get("source") or "").strip()
        if not src or not str(src).lower().endswith(".pdf"):
            continue
        try:
            page = m.get("page", 1)
            page = int(page) if (isinstance(page, int) and page >= 1) else 1
        except Exception:
            page = 1
        key = (os.path.normpath(src).lower(), page)
        if key in seen:
            continue
        seen.add(key)
        items.append({"src": src, "page": page, "base": os.path.basename(src)})
    return items

def render_pdf_grid(items, title="🖥️ Résultats", key_prefix="one"):
    """Grille d’aperçus PDF (expander toujours OUVERT)."""
    if not items:
        st.info("Aucun PDF trouvé pour cette requête.")
        return
    k = len(items)
    with st.expander(f"{title} ({k})", expanded=True):
        N_PER_ROW = 2
        for i in range(0, len(items), N_PER_ROW):
            cols = st.columns(N_PER_ROW)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(items):
                    with col: st.empty(); continue
                cur = items[idx]

                abs_path = _resolve_pdf_path(cur["src"])
                if not abs_path or not os.path.isfile(abs_path):
                    with col:
                        st.warning(f"[{idx+1}] Introuvable : {cur['base']}")
                    continue

                page_1based = cur["page"] if (isinstance(cur["page"], int) and cur["page"] >= 1) else 1
                page_idx0 = page_1based - 1

                with col:
                    st.markdown(f"**[{idx+1}] {cur['base']} — page {page_1based}**")
                    # Vignette
                    try:
                        png_bytes = _render_pdf_page_png(abs_path, page_idx0, zoom=2.2)
                        st.image(png_bytes, use_container_width=True)
                    except Exception as e:
                        st.error(f"Aperçu impossible : {e}")

                    # Deviner la page exacte via un snippet du dernier résultat
                    snippet = _find_first_snippet_for_source(cur["src"]) or ""
                    page_guess = page_1based
                    try:
                        pdf_bytes = _read_file_bytes(abs_path)
                        page_guess = guess_page_for_snippet(pdf_bytes=pdf_bytes, title="", snippet=snippet, ref="") or page_1based
                    except Exception:
                        page_guess = page_1based

                    btn_key = f"{key_prefix}_open_{idx+1}"
                    if st.button("🔗 Ouvrir le PDF", key=btn_key):
                        st.session_state["open_pdf"] = {
                            "path": abs_path,
                            "page": page_guess,
                            "base": cur["base"]
                        }


# =========================
#  États minimaux
# =========================
st.session_state.setdefault("docs_path", get_docs_path() or "")
st.session_state.setdefault("attest_path", get_attest_path() or "")
st.session_state.setdefault("active_corpus", get_last_corpus())  # "Mémoires" / "Attestations"
st.session_state.setdefault("last_docs", [])
st.session_state.setdefault("last_metas", [])
st.session_state.setdefault("last_dists", [])
st.session_state.setdefault("variant_offset", 0)


# =========================
#  Sidebar (menu déroulant + options)
# =========================
with st.sidebar:
    with st.expander("📂 Chemins de dossiers", expanded=False):
        # Corpus actif (pilote dossier + index)
        corpus = st.radio(
            "Chercher dans :",
            options=["Mémoires", "Attestations"],
            index=0 if st.session_state.active_corpus == "Mémoires" else 1,
            horizontal=True,
            key="corpus_radio",
        )
        set_last_corpus(corpus)

        # Chemin Mémoires
        mem_path = st.text_input(
            "Dossier Mémoires",
            value=st.session_state.docs_path,
            help="Ex: C:\\\\Users\\\\Moi\\\\DocsMémoire"
        )
        if mem_path.strip():
            if os.path.isdir(mem_path):
                set_docs_path(os.path.normpath(mem_path))
            else:
                st.warning("Chemin Mémoires invalide — on conserve l’éventuel chemin précédent valide.")
        else:
            st.caption("Aucun dossier Mémoires défini.")

        # Chemin Attestations
        att_path = st.text_input(
            "Dossier Attestations",
            value=st.session_state.attest_path,
            help="Ex: C:\\\\Users\\\\Moi\\\\DocsAttestations"
        )
        if att_path.strip():
            if os.path.isdir(att_path):
                set_attest_path(os.path.normpath(att_path))
            else:
                st.warning("Chemin Attestations invalide — on conserve l’éventuel chemin précédent valide.")
        else:
            st.caption("Aucun dossier Attestations défini.")

    st.divider()
    st.header("⚙️ Paramètres")
    topk = st.slider("Nombre de PDF à afficher (Top‑K)", 1, 10, 5)

    # Indexation spécifique au corpus sélectionné
    st.divider()
    st.header("🧱 Indexation")
    active_dir = get_active_base_dir()
    active_index = get_active_index_dir()
    st.caption(f"Reconstruire l’index pour **{st.session_state.active_corpus}** "
               f"({os.path.basename(active_index)})")
    if st.button("🔄 Mettre à jour la base"):
        if not active_dir or not os.path.isdir(active_dir):
            st.error("Définissez d’abord un dossier valide pour ce corpus.")
        else:
            prog = st.progress(0)
            try:
                py = sys.executable
                docs_arg  = os.path.normpath(active_dir)
                index_arg = os.path.normpath(active_index)
                os.makedirs(index_arg, exist_ok=True)
                # Vous avez déjà un script d’index (build_index) côté projet. On le réutilise.  ⤵
                cmd = [py, "-m", "app_logic.build_index", docs_arg, index_arg]

                re_total = re.compile(r"TOTAL_CHUNKS:(\d+)")
                re_prog  = re.compile(r"PROGRESS_CHUNKS:(\d+)")
                total = None
                processed = 0
                logs = []

                def render():
                    try:
                        prog.progress(int(processed * 100 / (total or 1)))
                    except Exception:
                        pass

                with st.spinner("Indexation…"):
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        cwd=ROOT
                    )
                    while True:
                        line = proc.stdout.readline()
                        if not line:
                            if proc.poll() is not None:
                                break
                            time.sleep(0.03)
                            continue
                        s = line.rstrip("\n")
                        logs.append(s)
                        m_total = re_total.search(s)
                        if m_total:
                            try: total = int(m_total.group(1))
                            except: total = None
                        m_prog = re_prog.search(s)
                        if m_prog:
                            try: processed = int(m_prog.group(1))
                            except: pass
                        render()

                    ret = proc.wait()
                    processed = max(processed, total or processed)
                    render()

                if ret == 0:
                    prog.progress(100)
                    st.success("Base mise à jour ✅")
                else:
                    st.error(f"Échec de la mise à jour (code {ret})")
                    with st.expander("Voir le log complet"):
                        st.code("\n".join(logs[-200:]))
            except Exception as e:
                st.error(f"Erreur: {e}")

# =========================
#  Zone Recherche (UN SEUL INPUT)
# =========================
st.subheader("Recherche dans vos documents")
with st.form(key="search_form", clear_on_submit=False):
    placeholder = "ex: tuile creuse, sanimur, toit…"
    query = st.text_input("Votre requête", placeholder=placeholder, key="query_text")
    submitted = st.form_submit_button("Rechercher 🔎")

if submitted and (query or "").strip():
    with st.spinner(f"Je cherche dans vos {st.session_state.active_corpus.lower()}…"):
        try:
            persist_dir = get_active_index_dir()
            raw_docs, raw_metas, raw_dists = tfidf_retrieve(persist_dir, query, topk * 8)
            docs, metas, dists = filter_unique_by_pdf(raw_docs, raw_metas, raw_dists, topk)
            # On mémorise pour pouvoir réafficher même après un rerun (clic sur bouton)
            st.session_state.last_docs  = docs
            st.session_state.last_metas = metas
            st.session_state.last_dists = dists
        except Exception as e:
            st.error(f"Erreur de recherche : {e}")
            st.session_state.last_docs  = []
            st.session_state.last_metas = []
            st.session_state.last_dists = []

elif submitted:
    st.warning("Saisissez une requête avant de lancer la recherche.")

# --- AFFICHAGE INCONDITIONNEL DES DERNIERS RÉSULTATS ---
# --- AFFICHAGE INCONDITIONNEL DES DERNIERS RÉSULTATS ---  ⟵ REMPLACE TOUT CE BLOC
items = build_pdf_items_from_metas(st.session_state.get("last_metas", []))
titre = f"🖥️ Résultats — {st.session_state.active_corpus}"

# Fenêtre : on n’affiche QUE 2 PDF à la fois
TOPK_WINDOW = 2

# Total uniques + sauvegarde pour l’état (utile pour (dés)activer les flèches)
total_uniques = len(items)
st.session_state.total_uniques = total_uniques

# si aucun résultat
if total_uniques == 0:
    render_pdf_grid([], title=titre, key_prefix="one")
else:
    # Corrige l'offset si nécessaire
    off = st.session_state.get("variant_offset", 0)
    if total_uniques <= TOPK_WINDOW:
        # Rien à paginer, offset inutile
        off = 0
        st.session_state.variant_offset = 0
        window = items  # affiche les 1 ou 2 items existants
    else:
        # Normalise et construit une fenêtre circulaire de 2 items
        off = off % total_uniques
        st.session_state.variant_offset = off
        take = min(TOPK_WINDOW, total_uniques)
        window = [items[(off + i) % total_uniques] for i in range(take)]

    # === Contrôles de variantes (flèches en haut) ===
    top_cols = st.columns([8, 1, 1, 8])
    disabled = (total_uniques <= TOPK_WINDOW)

    with top_cols[1]:
        if st.button("⬅️", disabled=disabled):
            st.session_state.variant_offset = (st.session_state.variant_offset - TOPK_WINDOW) % max(1, total_uniques)
            st.rerun()
    with top_cols[2]:
        if st.button("➡️", disabled=disabled):
            st.session_state.variant_offset = (st.session_state.variant_offset + TOPK_WINDOW) % max(1, total_uniques)
            st.rerun()

    # Rendu de la fenêtre courante (2 PDF max)
    # Astuce: on change key_prefix pour éviter les collisions de boutons
    key_prefix = f"page_{st.session_state.variant_offset}"
    render_pdf_grid(window, title=f"{titre} — {off+1}/{total_uniques}", key_prefix=key_prefix)


