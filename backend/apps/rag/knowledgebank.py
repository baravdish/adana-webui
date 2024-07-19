from abc import ABC, abstractmethod
from pydantic import Field
from typing import List, Union, Dict, Tuple, Callable, Optional
from langchain_core.documents import Document

class KnowledgeBase(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int, embedding_function: Callable) -> List['MyDocument']:
        pass

class MyDocument(Document):
    
    relevance_score: float = Field(default=0.0)

    def __init__(
        self,
        page_content: str,
        metadata: Optional[dict] = None,
        id: Optional[str] = None,
        relevance_score: float = 0.0,
    ):
        super().__init__(page_content=page_content, metadata=metadata or {}, id=id)
        self.relevance_score = relevance_score

    @property
    def source(self) -> Optional[str]:
        return self.metadata.get("source")

    @source.setter
    def source(self, value: str):
        self.metadata["source"] = value
    
    def __lt__(self, other: 'MyDocument') -> bool:
        return self.relevance_score < other.relevance_score

    def __repr__(self) -> str:
        return (f"MyDocument(content_preview='{self.page_content[:50]}...', "
                f"source='{self.source}', relevance_score={self.relevance_score})")

class DocumentStore:
    def __init__(self, documents: List[MyDocument]):
        self.documents = documents

    def __getitem__(self, index):
        return self.documents[index]

    def __len__(self):
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents)

    def __reversed__(self):
        return reversed(self.documents)

    def retrieve(self, query: str, k: int, embedding_function: Callable) -> List[MyDocument]:
        # Implement retrieval logic here
        # This is where you'd interact with get_rag_context()
        # For now, let's just return a simple filtered list
        retrieved_docs = [doc for doc in self.documents if query.lower() in doc.page_content.lower()]
        return sorted(retrieved_docs, key=lambda x: x.relevance_score, reverse=True)[:k]

class TextFileKnowledgeBase(KnowledgeBase):
    def retrieve(self, query: str, k: int, embedding_function: Callable) -> List[Document]:
        # Implementation for text files
        pass
    
class PDFKnowledgeBase(KnowledgeBase):
    def retrieve(self, query: str, k: int, embedding_function: Callable) -> List[Document]:
        # Implementation for PDF files
        pass
    
class DocxKnowledgeBase(KnowledgeBase):
    def retrieve(self, query: str, k: int, embedding_function: Callable) -> List[Document]:
        # Implementation for DOCX files
        pass
    
class ChromaDBKnowledgeBase(KnowledgeBase):
    def retrieve(self, query: str, k: int, embedding_function: Callable) -> List[Document]:
        # Implementation for ChromaDB collections
        pass
    
class CompositeKnowledgeBase(KnowledgeBase):
    def __init__(self, knowledge_bases: List[KnowledgeBase]):
        self.knowledge_bases = knowledge_bases

    def retrieve(self, query: str, k: int, embedding_function: Callable) -> List[Document]:
        all_docs = []
        for kb in self.knowledge_bases:
            all_docs.extend(kb.retrieve(query, k, embedding_function))
        # Combine and rank all retrieved documents
        return sorted(all_docs, key=lambda x: x.relevance_score, reverse=True)[:k]