# app_logic/rc_session.py
import os, re, io, json, math, time
from typing import List, Dict, Any, Tuple

import chromadb
from chromadb.utils import embedding_functions

# --- Dépendances internes ---
try:
    from .utils import load_openai_key
except Exception:
    def load_openai_key():
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if key:
            return key
        cfg_path = os.path.join(os.getcwd(), "config_user.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return (data.get("openai_api_key") or data.get("OPENAI_API_KEY") or "").strip()
            except Exception:
                pass
        return ""

# ---------- LECTURE DOCUMENTS ----------
def _read_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        import fitz  # PyMuPDF
        text = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for p in doc:
                text.append(p.get_text())
        return "\n".join(text)
    except Exception:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join([p.extract_text() or "" for p in reader.pages])

def _read_docx_bytes(docx_bytes: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(docx_bytes))
    return "\n".join([p.text for p in doc.paragraphs if p.text is not None])

def read_any(file_or_upload) -> str:
    if hasattr(file_or_upload, "read"):
        name = getattr(file_or_upload, "name", "").lower()
        data = file_or_upload.read()
    elif isinstance(file_or_upload, (str, os.PathLike)) and os.path.isfile(file_or_upload):
        name = str(file_or_upload).lower()
        with open(file_or_upload, "rb") as f:
            data = f.read()
    else:
        raise ValueError("read_any: entrée non reconnue ou fichier introuvable.")

    if name.endswith(".pdf"):
        return _read_pdf_bytes(data)
    if name.endswith(".docx"):
        return _read_docx_bytes(data)
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return data.decode("latin-1", errors="ignore")

# ---------- EXTRACTION CHAPITRES RC (regex simple) ----------
_CHAP_PAT = re.compile(
    r"(?im)^\s*(?:chapitre|section|article|\d+(?:\.\d+)*)(?:\s*[:\-\)]\s*|\s+)(.+?)\s*$"
)

def extract_rc_chapters(rc_text: str) -> List[str]:
    lines = [l.strip() for l in (rc_text or "").splitlines()]
    hits = []
    for l in lines:
        if not l:
            continue
        m = _CHAP_PAT.match(l)
        if m:
            title = re.sub(r"\s+", " ", m.group(0)).strip()
            hits.append(title)
    if not hits:
        for l in lines:
            if l.endswith(":") or (len(l) < 100 and l.isupper()):
                hits.append(l)
    seen, out = set(), []
    for h in hits:
        k = h.lower()
        if k not in seen:
            seen.add(k)
            out.append(h)
        if len(out) >= 50:
            break
    return out

# ---------- EMBEDDINGS OPENAI ----------
def _get_openai_embedder():
    key = load_openai_key()
    if not key:
        raise RuntimeError("Clé OpenAI introuvable. Renseigne OPENAI_API_KEY ou config_user.json.")
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=key,
        model_name="text-embedding-3-small"
    )

def _embed_texts(texts: List[str]) -> List[List[float]]:
    ef = _get_openai_embedder()
    return ef(texts)

def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    num = sum(x*y for x, y in zip(a, b))
    da = math.sqrt(sum(x*x for x in a))
    db = math.sqrt(sum(y*y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)

# ---------- DÉCOUPAGE CCTP PAR ARTICLE (embedding direct) ----------
_ARTICLE_SPLIT = re.compile(r"(?im)^\s*(?:article|art\.?)\s*([0-9A-Z\.\-]+)\s*[:\-\)]?\s*(.+)?$")

def chunk_cctp_for_matching(cctp_text: str) -> List[Dict[str, Any]]:
    lines = (cctp_text or "").splitlines()
    items = []
    current = {"article_no": None, "article_title": "", "text": []}

    def _push():
        if current["text"]:
            item = dict(current)
            item["text"] = "\n".join(current["text"]).strip()
            items.append(item)

    for l in lines:
        m = _ARTICLE_SPLIT.match(l)
        if m:
            if current["text"]:
                _push()
            current = {
                "article_no": m.group(1) or None,
                "article_title": (m.group(2) or "").strip(),
                "text": []
            }
        else:
            current["text"].append(l)
    _push()

    base_texts = []
    for it in items:
        head = f"ARTICLE {it.get('article_no') or ''} {it.get('article_title') or ''}".strip()
        clip = it["text"]
        if len(clip) > 3000:
            clip = clip[:3000]
        base_texts.append(f"{head}\n{clip}")
    if base_texts:
        embs = _embed_texts(base_texts)
        for it, e in zip(items, embs):
            it["emb"] = e
    return items

def rank_cctp_items_for_chapter(chapter_label: str, items: List[Dict[str, Any]], topn: int = 20) -> List[Dict[str, Any]]:
    if not items:
        return []
    q_emb = _embed_texts([chapter_label])[0]
    ranked = []
    for it in items:
        sc = _cosine(q_emb, it.get("emb") or [])
        ranked.append({
            "article_no": it.get("article_no"),
            "article_title": it.get("article_title"),
            "text": it.get("text"),
            "best_score": sc
        })
    ranked.sort(key=lambda r: r["best_score"], reverse=True)
    return ranked[:topn]

def summarize_by_article(ranked: List[Dict[str, Any]], max_per_article: int = 3, max_chars: int = 1200) -> List[Dict[str, Any]]:
    out = []
    for r in ranked:
        txt = (r.get("text") or "").strip()
        if len(txt) > max_chars:
            cut = txt[:max_chars]
            last_dot = cut.rfind(".")
            if last_dot > 300:
                cut = cut[:last_dot+1]
            txt = cut + " […]"
        out.append({**r, "summary": txt})
    return out[:max_per_article * 10]

# ---------- BRIEFS ----------
def build_detailed_brief(selected_rows: Dict[str, List[Dict[str, Any]]]) -> str:
    parts = []
    for chap, rows in selected_rows.items():
        parts.append(f"### {chap}\n")
        for r in rows:
            no = r.get("article_no") or "—"
            ti = r.get("article_title") or ""
            parts.append(f"- ARTICLE {no} {('– ' + ti) if ti else ''}\n{r.get('summary','')}\n")
        parts.append("")
    return "\n".join(parts).strip()

def build_chapter_brief(chap: str, rows: List[Dict[str, Any]]) -> str:
    buf = [f"Chapitre cible : {chap}", ""]
    for r in rows:
        no = r.get("article_no") or "—"
        ti = r.get("article_title") or ""
        buf.append(f"[ARTICLE {no} {('– ' + ti) if ti else ''}]")
        buf.append(r.get("summary") or r.get("text") or "")
        buf.append("")
    return "\n".join(buf)

# ---------- RECHERCHE DANS LES MÉMOIRES (ChromaDB) ----------
def _get_collection(persist_dir: str):
    ef = _get_openai_embedder()
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        col = client.get_collection("memoire_chunks", embedding_function=ef)
    except Exception:
        col = client.create_collection("memoire_chunks", embedding_function=ef)
    return col

def search_memoires_for_chapter(
    persist_dir: str,
    chapter_brief: str,
    n_results: int = 8,
    filter_doc_type: bool = False
) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
    col = _get_collection(persist_dir)
    where = {"doc_type": "memoire"} if filter_doc_type else None
    try:
        qr = col.query(
            query_texts=[chapter_brief],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
    except TypeError:
        qr = col.query(query_texts=[chapter_brief], n_results=n_results, where=where)

    docs = (qr.get("documents") or [[]])[0]
    metas = (qr.get("metadatas") or [[]])[0]
    dists = (qr.get("distances") or [[]])[0]

    norm_metas: List[Dict[str, Any]] = []
    for m in metas:
        m = dict(m or {})
        if "source_path" not in m and "source" in m:
            m["source_path"] = m.get("source")
        if "source_name" not in m and m.get("source_path"):
            try:
                import os as _os
                m["source_name"] = _os.path.basename(str(m["source_path"]))
            except Exception:
                m["source_name"] = str(m.get("source_path") or "")
        m.setdefault("source_pdf_bytes_path", "")
        try:
            p = int(m.get("page", 1))
            m["page"] = p if p >= 1 else 1
        except Exception:
            m["page"] = 1
        norm_metas.append(m)

    return docs, norm_metas, dists

def pick_top1_by_pdf(docs: List[str], metas: List[Dict[str, Any]], dists: List[float]):
    if not docs:
        return None, {}, None
    return docs[0], (metas[0] if metas else {}), (dists[0] if dists else None)

# =====================================================================
#                 🚀 PIPELINE LLM RC → CCTP → SYNTHÈSE (MODE TEXTE)
# =====================================================================
SYSTEM_SCAN = """Tu es un extracteur de clauses techniques dans un CCTP.
Tu reçois un intitulé de chapitre RC (thème) et un extrait de CCTP.
Tâche:
- Dire si l'extrait contient des clauses PERTINENTES (oui/non) pour ce thème.
- Si oui, extraire des obligations techniques en phrases courtes (sans blabla), en conservant le sens.
- Conserver références (ARTICLE, §) si visibles.
Réponds en JSON STRICT:
{"relevant": true/false, "excerpts": ["clause...", "..."], "why": "raison", "score": 0.0}
"""

USER_SCAN_TMPL = """RC_CHAPTER_TITLE: "{chapter}"

CCTP_CHUNK_BEGIN
{chunk}
CCTP_CHUNK_END

Rappels:
- Si aucune clause utile: relevant=false, excerpts=[].
- Sinon: liste d'excerpts atomiques, sans paraphrases inutiles.
- JSON strict uniquement.
"""

SYSTEM_SYNTH = """Tu synthétises des exigences techniques à partir de clauses brutes extraites d'un CCTP.
Objectif: produire 5 à 12 puces ultra concises d'obligations/tolérances/normes/contrôles/mise en œuvre.
Retour JSON strict:
{"constraints_synthesis": ["- ...", "- ..."], "coverage": 0.0}
"""

USER_SYNTH_TMPL = """CHAPTER_TITLE: "{chapter}"
CLAUSES:
{clauses}
"""

def _openai_client(explicit_key: str = ""):
    from openai import OpenAI
    key = (explicit_key or load_openai_key() or "").strip()
    if not key:
        raise RuntimeError("Clé OpenAI manquante. Renseigne OPENAI_API_KEY ou config_user.json.")
    return OpenAI(api_key=key)

def _chat_json(client, system: str, user: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    txt = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(txt)
    except Exception:
        try:
            import orjson
            return orjson.loads(txt)
        except Exception:
            return {}

def _chunk_text(s: str, max_chars: int = 12000, overlap: int = 800) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    if len(s) <= max_chars:
        return [s]
    out, i = [], 0
    while i < len(s):
        out.append(s[i:i+max_chars])
        i += max_chars - overlap
    return out

def cctp_synthesize_for_chapters_llm(
    *,
    rc_chapter_titles: List[str],
    cctp_text: str,
    openai_key: str = "",
    model_reason: str = "gpt-4o-mini",
    model_synth: str = "gpt-4o-mini",
    max_chunks_per_chapter: int = 60
) -> Dict[str, Dict[str, Any]]:
    client = _openai_client(openai_key)
    chunks = _chunk_text(cctp_text, 12000, 800)
    result: Dict[str, Dict[str, Any]] = {}

    for chapter in rc_chapter_titles:
        clauses: List[str] = []
        sources: List[Dict[str, Any]] = []
        for i, ch in enumerate(chunks):
            if i >= max_chunks_per_chapter:
                break
            data = _chat_json(
                client,
                SYSTEM_SCAN,
                USER_SCAN_TMPL.format(chapter=chapter, chunk=ch[:18000]),
                model=model_reason
            )
            if data.get("relevant") and data.get("excerpts"):
                for ex in data["excerpts"]:
                    ex = (ex or "").strip()
                    if ex:
                        clauses.append(ex)
                sources.append({
                    "chunk_index": i,
                    "why": (data.get("why") or "").strip(),
                    "score": float(data.get("score") or 0.0)
                })

        seen, dedup = set(), []
        for ex in clauses:
            k = ex.lower().strip()
            if k not in seen:
                seen.add(k)
                dedup.append(ex)

        joined = "\n".join(f"- {c}" for c in dedup[:400])
        syn = _chat_json(
            client, SYSTEM_SYNTH,
            USER_SYNTH_TMPL.format(chapter=chapter, clauses=joined),
            model=model_synth
        )
        constraints = syn.get("constraints_synthesis") or []
        cov = float(syn.get("coverage") or 0.0)
        result[chapter] = {
            "constraints_synthesis": constraints,
            "coverage": cov,
            "sources": sources
        }

    return result

# =====================================================================
#                 🚀 MODE FICHIER (Assistants v2 + file_search)
# =====================================================================

from openai import OpenAI as _OAClient

_ASSISTANT_SYSTEM = """Tu es un extracteur de clauses techniques dans un CCTP.
Objectif par chapitre RC: produire 5 à 12 puces d'exigences concrètes (obligations, tolérances, normes, contrôles, mise en œuvre).
Retour JSON strict:
{"constraints_synthesis": ["- ...", "- ..."], "coverage": 0.0}
Ne renvoie rien d'autre que le JSON.
"""

def _assistant_client(explicit_key: str = "") -> _OAClient:
    key = (explicit_key or load_openai_key() or "").strip()
    if not key:
        raise RuntimeError("Clé OpenAI manquante. Renseigne OPENAI_API_KEY ou config_user.json.")
    return _OAClient(api_key=key)

def upload_to_openai(file_bytes: bytes, file_name: str, api_key: str = "") -> str:
    client = _assistant_client(api_key)
    up = client.files.create(
        file=(file_name or "CCTP.pdf", io.BytesIO(file_bytes), "application/pdf"),
        purpose="assistants",
    )
    return up.id



# Cache assistant pour éviter de le recréer à chaque appel
_ASSISTANT_CACHE: dict[str, str] = {}

def _get_or_create_assistant_id(client, model: str) -> str:
    """
    Crée une seule fois un Assistant avec file_search, puis réutilise son ID.
    """
    if model in _ASSISTANT_CACHE:
        return _ASSISTANT_CACHE[model]

    asst = client.beta.assistants.create(
        name="TMH – Extracteur CCTP",
        model=model,
        instructions=_ASSISTANT_SYSTEM,
        tools=[{"type": "file_search"}],
    )
    _ASSISTANT_CACHE[model] = asst.id
    return asst.id


def _run_assistant_json_for_chapter(
    *, client: _OAClient, assistant_id: str, file_id: str, chapter_title: str
) -> dict:
    # 1) Créer un thread
    thread = client.beta.threads.create()

    # 2) Poster le message utilisateur avec le PDF attaché
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=(
            "Analyse le CCTP joint et extrais uniquement les clauses utiles au chapitre suivant.\n"
            f'CHAPITRE CIBLE: "{chapter_title}".\n'
            "Réponds STRICTEMENT au format JSON demandé."
        ),
        attachments=[{"file_id": file_id, "tools": [{"type": "file_search"}]}],
    )

    # 3) Lancer le run avec assistant_id
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

    # 4) Attendre la fin
    import time as _t
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run.status in ("completed", "failed", "cancelled", "expired"):
            break
        _t.sleep(0.5)

    if run.status != "completed":
        raise RuntimeError(f"Assistant run non terminé (statut={run.status}).")

    # 5) Récupérer la réponse texte
    msgs = client.beta.threads.messages.list(thread_id=thread.id, order="desc", limit=5)
    for m in msgs.data:
        if m.role == "assistant":
            txt = "\n".join(
                c.text.value for c in m.content if c.type == "text" and getattr(c, "text", None)
            ).strip()
            if txt:
                try:
                    return json.loads(txt)
                except Exception:
                    try:
                        import orjson
                        return orjson.loads(txt)
                    except Exception:
                        return {"constraints_synthesis": [txt], "coverage": 0.0}
    return {"constraints_synthesis": [], "coverage": 0.0}

def cctp_synthesize_for_chapters_file(
    *,
    rc_chapter_titles: List[str],
    cctp_bytes: bytes,
    cctp_name: str,
    openai_key: str = "",
    model: str = "gpt-4o-mini"
) -> Dict[str, Dict[str, Any]]:
    if not cctp_bytes:
        raise ValueError("CCTP vide.")
    client = _assistant_client(openai_key)
    file_id = upload_to_openai(cctp_bytes, cctp_name or "CCTP.pdf", openai_key)
    assistant_id = _get_or_create_assistant_id(client, model)

    results: Dict[str, Dict[str, Any]] = {}
    for chapter in rc_chapter_titles:
        data = _run_assistant_json_for_chapter(
            client=client,
            assistant_id=assistant_id,   # ✅ ici on passe l’assistant_id
            file_id=file_id,
            chapter_title=chapter,
        )
        constraints = data.get("constraints_synthesis") or []
        cov = float(data.get("coverage") or 0.0)
        results[chapter] = {
            "constraints_synthesis": constraints,
            "coverage": cov,
            "sources": []
        }
    return results
