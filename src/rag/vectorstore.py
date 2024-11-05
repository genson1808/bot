from typing import Union
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class VectorDB:
    def __init__(self,
                 documents=None,
                 vector_db: Union[Chroma, FAISS] = None,
                 embedding=None,
                 ) -> None:
        # Initialize vector_db properly
        if vector_db is None:
            vector_db = Chroma()  # Instantiate Chroma if it's not provided
        self.vector_db = vector_db

        self.embedding = self._initialize_embedding()  # Initialize embedding first
        self.db = self._build_db(documents)

    def _initialize_embedding(self):
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    def _build_db(self, documents):
        db = self.vector_db.from_documents(documents=documents, embedding=self.embedding)
        return db

    def get_retriever(self,
                      search_type: str = "similarity",
                      search_kwargs: dict = {"k": 10}):
        retriever = self.db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        return retriever
