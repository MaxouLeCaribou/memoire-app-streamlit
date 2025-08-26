import tiktoken

# x3 par rapport à 600
CHUNK_TOKENS = 600
OVERLAP_TOKENS = 150  # ~8–10%

_enc = tiktoken.get_encoding("cl100k_base")

def chunk_text(text: str):
    toks = _enc.encode(text)
    i, out = 0, []
    while i < len(toks):
        sub = toks[i:i+CHUNK_TOKENS]
        out.append(_enc.decode(sub))
        i += max(1, CHUNK_TOKENS - OVERLAP_TOKENS)
    return out
