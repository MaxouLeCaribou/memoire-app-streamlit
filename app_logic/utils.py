import os

def load_openai_key(path: str = None) -> str:
    if path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(base, ".."))
        path = os.path.join(root, "config", "openai_key.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Clé introuvable: {path}")
    with open(path, "r", encoding="utf-8") as f:
        key = f.read().strip()
    if not key:
        raise ValueError("Fichier clé vide.")
    return key
