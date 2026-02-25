"""Quick smoke test — verify ChromaDB index is populated and searchable."""
from src.indexing.embedder import embed_documents
from src.indexing.vector_store import get_client, get_collection, similarity_search

client = get_client()
col = get_collection(client)
print("Total docs in ChromaDB:", col.count())

query = "cheap South Indian breakfast place"
vec = embed_documents([query])[0]
results = similarity_search(col, vec, top_k=5)

for r in results:
    m = r["metadata"]
    print(f"  {m['name']} | {m['location']} | rate={m['rate']} | cost={m['approx_cost']}")
