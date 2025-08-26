import os, sys
import chromadb
from chromadb.utils import embedding_functions
from app_logic.utils import load_openai_key

def main():
    persist_dir = sys.argv[1] if len(sys.argv)>1 else os.path.join(os.getcwd(),"chroma_index")
    query = sys.argv[2] if len(sys.argv)>2 else "Méthodologie de sécurisation en site occupé avec accès piétons"
    topk = int(sys.argv[3]) if len(sys.argv)>3 else 5

    OPENAI_API_KEY = load_openai_key()
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )

    client = chromadb.PersistentClient(path=persist_dir)
    coll = client.get_collection("memoire_chunks", embedding_function=ef)

    res = coll.query(query_texts=[query], n_results=topk, include=["documents","metadatas","distances"])
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    print(f"Query: {query}\nTop {topk} résultats:\n")
    for i,(d,m,dist) in enumerate(zip(docs, metas, dists),1):
        print(f"#{i} dist={dist:.4f}")
        print(f"source: {m.get('source')} | para_idx: {m.get('para_idx')}")
        print(d[:500].replace("\n"," ")+"...\n")

if __name__=="__main__":
    main()
