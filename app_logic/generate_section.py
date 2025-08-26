import os, sys, argparse
import chromadb
from chromadb.utils import embedding_functions
from app_logic.utils import load_openai_key
from docx import Document
from openai import OpenAI

def retrieve(persist_dir: str, query: str, topk: int = 5):
    """Retourne (docs, metas, dists) des top-k extraits depuis Chroma."""
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=load_openai_key(),
        model_name="text-embedding-3-small"
    )
    client = chromadb.PersistentClient(path=persist_dir)
    coll = client.get_collection("memoire_chunks", embedding_function=ef)

    res = coll.query(query_texts=[query], n_results=topk, include=["documents","metadatas","distances"])
    return res["documents"][0], res["metadatas"][0], res["distances"][0]

def print_hits(docs, metas, dists):
    print("\n=== TOP PASSAGES (pour vérification) ===\n")
    for i, (d, m, dist) in enumerate(zip(docs, metas, dists), 1):
        print(f"#{i} dist={dist:.4f}")
        print(f"source: {m.get('source')} | para_idx: {m.get('para_idx')}")
        print(d[:500].replace("\n"," ") + "...\n")

def build_prompt(query: str, docs, metas):
    # On prépare un contexte numéroté [1], [2], ...
    snippets = []
    for i, (d, m) in enumerate(zip(docs, metas), 1):
        src = m.get("source")
        snippets.append(f"[{i}] {d}\n(SOURCE: {src})")
    ctx = "\n\n".join(snippets)

    system = (
        "Tu es un rédacteur de mémoire technique. "
        "Tu écris de manière claire, structurée (titres courts, listes si utile), "
        "et tu RESTES STRICTEMENT dans les informations fournies par le contexte. "
        "Si une info n'est pas dans le contexte, tu dis que ce n'est pas précisé."
    )

    user = (
        f"Question / Consigne:\n{query}\n\n"
        f"Contextes à utiliser (cites-les sous forme [1], [2]... quand tu réutilises une info) :\n{ctx}\n\n"
        "Rédige une section de mémoire technique en 1 à 4 paragraphes. "
        "Commence par un court titre. Termine par une liste 'Références internes utilisées' avec [numéro] + chemin SOURCE."
    )
    return system, user

def save_docx(text: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    doc.save(out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("persist_dir", help="Dossier de l'index Chroma (ex: ./chroma_index)")
    parser.add_argument("query", help="Question/consigne (ex: 'Sécurité chantier en site occupé')")
    parser.add_argument("topk", nargs="?", type=int, default=5, help="Nombre d'extraits (défaut 5)")
    parser.add_argument("--out", help="Chemin DOCX de sortie (ex: ./resultats/section_securite.docx)")
    args = parser.parse_args()

    # 1) Récupérer les passages
    docs, metas, dists = retrieve(args.persist_dir, args.query, args.topk)

    # 2) Afficher les hits (on garde cette étape)
    print_hits(docs, metas, dists)

    # 3) Construire le prompt et appeler OpenAI
    system, user = build_prompt(args.query, docs, metas)
    client = OpenAI(api_key=load_openai_key())
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user}
        ]
    )
    text = resp.choices[0].message.content

    print("\n=== SECTION GÉNÉRÉE ===\n")
    print(text)

    # 4) Option de sauvegarde DOCX
    if args.out:
        save_docx(text, args.out)
        print(f"\n[OK] Section enregistrée : {args.out}")

if __name__ == "__main__":
    main()
