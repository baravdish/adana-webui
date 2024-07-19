# In backend/apps/rag/reranking.py
from typing import List
from backend.apps.rag.knowledgebank import MyDocument  # Adjust import path as needed


class ReRanker:
    def __init__(self, strategy='default'):
        self.strategy = strategy

    def rerank_documents(self, query: str, documents: List[MyDocument]) -> List[MyDocument]:
        if self.strategy == 'default':
            return sorted(documents, key=lambda doc: self.default_scoring(query, doc), reverse=True)
        # Add other strategies as needed

    def default_scoring(self, query: str, doc: MyDocument) -> float:
        # Implement a basic scoring logic
        # This is a simple example; you might want to use more sophisticated methods
        query_words = set(query.lower().split())
        doc_words = set(doc.page_content.lower().split())
        common_words = query_words.intersection(doc_words)
        
        # Calculate a simple relevance score based on word overlap
        score = len(common_words) / len(query_words) if query_words else 0
        
        # Combine with the existing relevance score
        return score * 0.5 + doc.relevance_score * 0.5
