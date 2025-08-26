# app_logic/build_index.py — Indexeur Chroma SANS embeddings (compatible TF‑IDF côté app)
import os, sys, re, json
import math
import chromadb
import fitz  # PyMuPDF

import io
from PIL import Image
import pytesseract


COLLECTION_NAME = "memoire_chunks"  # ← doit rester identique à l'app

def log_total(n: int):
    print(f"TOTAL_CHUNKS:{n}", flush=True)

def log_progress(n: int):
    print(f"PROGRESS_CHUNKS:{n}", flush=True)

def get_collection_no_embedding(persist_dir: str):
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        # Création/ouverture SANS embedding_function
        return client.get_or_create_collection(name=COLLECTION_NAME)
    except ValueError as e:
        # Conflit de config (collection créée autrefois avec une embedding_function)
        msg = str(e).lower()
        if "embedding function already exists" in msg or "embedding function conflict" in msg:
            try:
                client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            return client.get_or_create_collection(name=COLLECTION_NAME)
        raise


def _ocr_page(page) -> str:
    """OCR sur l'image de la page avec Tesseract (zoom x2 pour la qualité)."""
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        # Mets lang="fra" si tes attestations sont FR (après avoir installé le pack fra)
        txt = pytesseract.image_to_string(img, lang="fra")
        return re.sub(r"\s+", " ", txt).strip()
    except Exception:
        return ""



def chunk_pdf(abs_path: str, max_chars: int = 1200, overlap: int = 120):
    """Découpe simple par page avec fallback OCR si la page n’a pas de texte."""
    doc = fitz.open(abs_path)
    try:
        for page_i in range(doc.page_count):
            page = doc.load_page(page_i)
            text = page.get_text("text") or ""
            text = re.sub(r"\s+", " ", text).strip()

            # Fallback OCR si pas de texte
            if not text:
                text = _ocr_page(page)

            if not text:
                continue  # page vraiment vide

            if len(text) <= max_chars:
                yield (page_i + 1), text
            else:
                start = 0
                while start < len(text):
                    end = min(len(text), start + max_chars)
                    yield (page_i + 1), text[start:end]
                    if end == len(text):
                        break
                    start = max(0, end - overlap)
    finally:
        doc.close()


def iter_pdfs(root_dir: str):
    exts = {".pdf"}
    for r, _, files in os.walk(root_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                yield os.path.normpath(os.path.join(r, f))

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m app_logic.build_index <docs_dir> <persist_dir>")
        sys.exit(2)

    docs_dir   = sys.argv[1]
    persist_dir= sys.argv[2]

    print(f"Dossier documents : {docs_dir}")
    print(f"Chroma persist_dir: {persist_dir}")
    os.makedirs(persist_dir, exist_ok=True)

    # 1) Collecte des PDF
    pdfs = [p for p in iter_pdfs(docs_dir)]
    if not pdfs:
        log_total(0)
        sys.exit(0)

    # 2) Comptage “approx” des chunks
    approx_total = 0
    for p in pdfs:
        try:
            with fitz.open(p) as d:
                for _ in range(d.page_count):
                    approx_total += 1  # estimation minimale (au moins 1 chunk/page)
        except Exception:
            pass
    log_total(approx_total)

    # 3) Ouverture collection SANS embeddings
    coll = get_collection_no_embedding(persist_dir)

    # 4) Ajout des chunks (documents + metadatas + ids) — SANS embeddings
    processed = 0
    batch_docs, batch_metas, batch_ids = [], [], []
    BID = 0

    for pdf_path in pdfs:
        base = os.path.basename(pdf_path)
        added_for_this_pdf = 0
        try:
            for page_1b, chunk in chunk_pdf(pdf_path):
                BID += 1
                doc_id = f"{base}::{page_1b}::{BID}"
                batch_docs.append(chunk)
                batch_metas.append({
                    "source": pdf_path,
                    "page": int(page_1b),
                    "source_name": base,
                })
                batch_ids.append(doc_id)
                added_for_this_pdf += 1

                # Flush par paquets
                if len(batch_docs) >= 100:
                    coll.add(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)
                    processed += len(batch_docs)
                    log_progress(processed)
                    batch_docs, batch_metas, batch_ids = [], [], []
        except Exception as e:
            print(f"[WARN] Impossible de traiter {pdf_path}: {e}", flush=True)

        print(f"[INFO] {base}: {added_for_this_pdf} chunks ajoutés", flush=True)


    # Dernier flush
    if batch_docs:
        coll.add(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)
        processed += len(batch_docs)
        log_progress(processed)

    # Fin propre
    sys.stdout.flush()

if __name__ == "__main__":
    main()
