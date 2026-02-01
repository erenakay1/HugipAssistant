"""
Vectorstore Service
Pinecone vectorstore yönetimi
"""
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List
from src.core.config import get_settings

class VectorStoreService:
    """Pinecone vectorstore servisi"""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.settings.OPENAI_API_KEY
        )
        
        # Vectorstore
        self.vectorstore = PineconeVectorStore(
            index_name=self.settings.PINECONE_INDEX_NAME,
            embedding=self.embeddings
        )
    
    def retrieve_documents(
        self, 
        query: str, 
        k: int = None,
        use_mmr: bool = True
    ) -> List[Document]:
        """
        Pinecone'dan ilgili dökümanları getirir
        
        Args:
            query: Arama sorgusu
            k: Kaç döküman getirilecek (default: settings.RETRIEVAL_K)
            use_mmr: MMR (Maximum Marginal Relevance) kullan - çeşitlilik için
            
        Returns:
            List[Document]: Retrieved dökümanlar
        """
        k = k or self.settings.RETRIEVAL_K
        if use_mmr:
            # MMR: Relevance + Diversity
            # fetch_k: İlk kaç dökümanı çek (sonra k tanesini seç)
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": k * 3,  # 3x daha fazla getir, sonra en iyilerini seç
                    "lambda_mult": 0.7  # 0.7 = relevance ağırlıklı, 0.3 = diversity
                }
            )
        else:
            # Standard similarity search
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            )
        return retriever.invoke(query)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = None
    ):
        """
        Score'larla birlikte arama
        Debugging ve kalite kontrolü için
        """
        k = k or self.settings.RETRIEVAL_K
        return self.vectorstore.similarity_search_with_score(query, k=k)