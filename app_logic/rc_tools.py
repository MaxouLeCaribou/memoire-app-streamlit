# app_logic/rc_tools.py
from __future__ import annotations
import io, json, os, re, subprocess, sys, uuid
from typing import List, Dict, Any, Tuple, Optional

# -----------------------------
# Parsing de fichiers (bytes -> texte)
# -----------------------------
def read_file_text_and_bytes(name: str, data: bytes) -> Tuple[str, bytes]:
    """
    Retourne (texte_brut, bytes) pour PDF/DOCX/TXT/MD/RTF, à partir d'un nom de fichier et de ses bytes.
    Ne dépend pas de Streamlit.
    """
    lname = (name or "").lower()
    if lname.endswith(".pdf"):
        # PDF -> texte via PyMuPDF puis bytes (fallback PyPDF2)
        try:
            import fitz  # PyMuPDF
            text_parts = []
            with fitz.open(stream=data, filetype="pdf") as doc:
                for p in doc:
                    text_parts.append(p.get_text())
            return "\n".join(text_parts), data
        except Exception:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(io.BytesIO(data))
                text = "\n".join([(p.extract_text() or "") for p in reader.pages])
                return text, data
            except Exception:
                return "", data

    elif lname.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(data))
            text = "\n".join([p.text for p in doc.paragraphs if p.text])
            return text, data
        except Exception:
            return "", data

    # txt/md/rtf: décodage simple
    try:
        return data.decode("utf-8", errors="ignore"), data
    except Exception:
        return data.decode("latin-1", errors="ignore"), data


# -----------------------------
# Extraction LLM des chapitres "mémoire"
# -----------------------------
def extract_mem_chapters_llm(
    rc_text: str,
    openai_key: str,
    model: str = "gpt-4o-mini",
    max_chars: int = 180_000,
) -> List[Dict[str, str]]:
    """
    Utilise l'API OpenAI pour extraire uniquement les sections RC où le mémoire technique est exigé.
    Retour: liste de dicts {title, ref, why, snippet}.
    """
    from openai import OpenAI
    client = OpenAI(api_key=(openai_key or "").strip())
    if not rc_text:
        return []

    system_prompt = (
        "Tu es un assistant d’analyse d’appels d’offres. "
        "Ta mission est d’identifier uniquement la ou les sections du RC/RPAO "
        "où il est explicitement demandé au candidat de fournir un mémoire technique "
        "(ou mémoire justificatif, note méthodologique, présentation technique), "
        "et de lister précisément ce que ce mémoire doit contenir. "
        "Ignore toutes les autres informations du RC (administratif, financier, juridique, critères d’attribution, etc.).\n\n"
        "Réponds au format JSON compact avec la clé \"mem_chapters\" (liste d’objets):\n"
        "[\n"
        "  {\"title\":\"<intitulé exact du chapitre ou sous-chapitre>\", "
        "\"ref\":\"<numéro ou section s’il existe>\", "
        "\"why\":\"<raison courte de la sélection>\", "
        "\"snippet\":\"<extrait exact du RC qui indique le contenu du mémoire>\"}\n"
        "]\n"
        "Si aucun passage ne correspond : {\"mem_chapters\": []}"
    )

    rc_text_cut = (rc_text or "")[:max_chars]
    user_prompt = (
        "Voici le texte intégral du RC/RPAO (UTF-8). Extrait uniquement les parties pertinentes au mémoire technique "
        "selon les règles ci-dessus.\nRC:\n<<<\n" + rc_text_cut + "\n>>>"
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    )
    raw = (resp.choices[0].message.content or "").strip()

    # Parsing robuste d'un bloc JSON potentiellement entouré de texte
    parsed = {}
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(raw[start:end+1])
    except Exception:
        parsed = {}

    mem_chapters = parsed.get("mem_chapters") if isinstance(parsed, dict) else None
    if not isinstance(mem_chapters, list):
        return []

    out: List[Dict[str, str]] = []
    for it in mem_chapters:
        if not isinstance(it, dict):
            continue
        title = (it.get("title") or "").strip()
        if not title:
            continue
        out.append({
            "title": title,
            "ref": (it.get("ref") or "").strip(),
            "why": (it.get("why") or "").strip(),
            "snippet": (it.get("snippet") or "").strip()
        })
    return out


# -----------------------------
# Localisation de page dans un PDF
# -----------------------------
def guess_page_for_snippet(pdf_bytes: bytes, title: str, snippet: str, ref: str = "") -> int:
    """
    Trouve la page en cherchant les mots-clés du titre, snippet et éventuellement la référence.
    """
    try:
        import fitz  # PyMuPDF
    except Exception:
        return 1

    import re, unicodedata

    def _normalize(s: str) -> str:
        s = unicodedata.normalize("NFKD", s or "")
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    # Mots-clés depuis titre + snippet + ref
    text_search = f"{title} {snippet} {ref}".strip()
    keywords = [w for w in _normalize(text_search).split() if len(w) >= 4]

    if not keywords:
        return 1

    best_page = 1
    best_score = 0
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for i, page in enumerate(doc, start=1):
                page_text = _normalize(page.get_text())
                score = sum(1 for kw in keywords if kw in page_text)
                if score > best_score:
                    best_score = score
                    best_page = i
    except Exception:
        pass

    return best_page




# -----------------------------
# Gestion des PDFs temporaires et ouverture externe
# -----------------------------
def _unique_pdf_path(tmp_dir: str) -> str:
    return os.path.join(tmp_dir, f"rc_{uuid.uuid4().hex}.pdf")

def save_temp_pdf(pdf_bytes: bytes, root_dir: str, suggested_name: Optional[str] = None) -> str:
    """
    Sauvegarde le PDF dans {root_dir}/tmp avec un nom unique (évite les verrous Windows).
    - Si suggested_name est donné, on tente d’abord ce nom, sinon UUID direct.
    - En cas de PermissionError/accès refusé, on retente avec un UUID.
    """
    tmp_dir = os.path.join(root_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # 1) Essai sur suggested_name
    if suggested_name:
        path_try = os.path.join(tmp_dir, suggested_name)
        try:
            with open(path_try, "wb") as f:
                f.write(pdf_bytes)
            return path_try
        except Exception:
            pass

    # 2) Essais avec UUID (quelques tentatives)
    for _ in range(3):
        path_try = _unique_pdf_path(tmp_dir)
        try:
            with open(path_try, "wb") as f:
                f.write(pdf_bytes)
            return path_try
        except Exception:
            continue

    # 3) Dernier recours
    fallback = os.path.join(tmp_dir, f"rc_fallback_{uuid.uuid4().hex}.pdf")
    with open(fallback, "wb") as f:
        f.write(pdf_bytes)
    return fallback


def open_pdf_external(
    path: str,
    page: int = 1,
    custom_viewer: Optional[str] = None
) -> bool:
    """
    Ouvre le PDF à la page donnée.
    Ordre d'essai si aucun viewer donné :
      1) Edge  2) Chrome  3) Acrobat  4) Sumatra  5) webbrowser (URI#page=)  6) association OS
    """
    import sys, os, subprocess, glob, webbrowser

    page = int(page) if isinstance(page, int) and page >= 1 else 1
    viewer = (custom_viewer or "").strip()

    def _as_uri(p: str) -> str:
        return "file:///" + os.path.abspath(p).replace("\\", "/")

    def _try(cmd: list) -> bool:
        try:
            subprocess.Popen(cmd)
            return True
        except Exception:
            return False

    def _open_edge(uri: str) -> bool:
        for p in [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]:
            if os.path.isfile(p):
                return _try([p, f"{uri}#page={page}"])
        return False

    def _open_chrome(uri: str) -> bool:
        for p in [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]:
            if os.path.isfile(p):
                return _try([p, f"{uri}#page={page}"])
        return False

    def _open_acrobat(path_exe: str, pdf: str) -> bool:
        return _try([path_exe, "/A", f"page={page}", pdf])

    def _open_sumatra(path_exe: str, pdf: str) -> bool:
        return _try([path_exe, "-page", str(page), pdf])

    # 1) viewer explicite
    if viewer:
        v = os.path.basename(viewer).lower()
        uri = _as_uri(path)
        if "msedge" in v or "edge" in v:
            return _try([viewer, f"{uri}#page={page}"])
        if "chrome" in v:
            return _try([viewer, f"{uri}#page={page}"])
        if "acrord32" in v or "acrobat" in v or "reader" in v:
            return _open_acrobat(viewer, path)
        if "sumatra" in v:
            return _open_sumatra(viewer, path)
        return _try([viewer, path])  # inconnu

    uri = _as_uri(path)

    # 2) Edge → 3) Chrome
    if sys.platform.startswith("win"):
        if _open_edge(uri):
            return True
        if _open_chrome(uri):
            return True
        # 4) Acrobat
        for pat in [
            r"C:\Program Files\Adobe\Acrobat Reader*\Reader\AcroRd32.exe",
            r"C:\Program Files (x86)\Adobe\Acrobat Reader*\Reader\AcroRd32.exe",
        ]:
            for exe in glob.glob(pat):
                if os.path.isfile(exe) and _open_acrobat(exe, path):
                    return True
        # 5) Sumatra
        for exe in [
            r"C:\Program Files\SumatraPDF\SumatraPDF.exe",
            r"C:\Program Files (x86)\SumatraPDF\SumatraPDF.exe",
        ]:
            if os.path.isfile(exe) and _open_sumatra(exe, path):
                return True
        # 6) webbrowser (navigateur par défaut) avec #page=
        if webbrowser.open(f"{uri}#page={page}", new=1):
            return True
        # 7) dernier recours association OS (peut ignorer la page)
        try:
            os.startfile(os.path.abspath(path))  # type: ignore[attr-defined]
            return True
        except Exception:
            return False

    elif sys.platform == "darwin":
        if _try(["open", "-a", "Microsoft Edge", f"{uri}#page={page}"]):
            return True
        if _try(["open", "-a", "Google Chrome", f"{uri}#page={page}"]):
            return True
        import webbrowser
        if webbrowser.open(f"{uri}#page={page}", new=1):
            return True
        return _try(["open", os.path.abspath(path)])

    else:
        # Linux
        if _try(["xdg-open", f"{uri}#page={page}"]):
            return True
        import webbrowser
        if webbrowser.open(f"{uri}#page={page}", new=1):
            return True
        return _try(["xdg-open", os.path.abspath(path)])
