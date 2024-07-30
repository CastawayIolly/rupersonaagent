from langchain.vectorstores import faiss


class FaissRetrieverCosine:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        # Используем IndexFlatIP для косинусной близости
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query_embedding, k=5):
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        _, indices = self.index.search(query_embedding, k)
        return indices[0]
