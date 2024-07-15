"""
license: MIT
description: A simple RAG (Retrieval-Augmented Generation) pipeline for Open WebUI
"""

from typing import List, Union, Generator, Iterator
# from schemas import OpenAIChatMessage
# import numpy as np
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
import os
import logging
import nltk
nltk.download('punkt')

class DocumentLoadError(Exception):
    """Custom exception for document loading errors"""
    pass


class Pipeline:
    def __init__(self):
        """
        Initialize the Pipeline with necessary attributes for document processing and RAG operations.
        """

        self.documents = []
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = None
        self.data_dir = "./data/"
        self.document_dir = os.path.join(self.data_dir, "documents")
        self.batch_size = 1000
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding
        self.max_tokens = 2048  # Adjust based on your needs and model limits
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    async def on_startup(self):
        # This function is called when the server is started.
        # We'll implement document loading and indexing here
        # self.load_documents()
        # self.index_documents()
        
        self.initialize()

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        # We can add any cleanup code here if needed
        pass
    
    def initialize(self):
        """
        Initialize the pipeline by loading and indexing documents.
        """
        try:
            self.load_documents()
            self.index_documents()
            self.logger.info("Pipeline initialization completed successfully.")
        except Exception as e:
            self.logger.error(f"Error during pipeline initialization: {str(e)}")
            

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.tokenizer.encode(text))
    
    def truncate_to_token_limit(self, text: str, limit: int) -> str:
        """Truncate text to a specified token limit."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= limit:
            return text
        return self.tokenizer.decode(tokens[:limit])

    
    def load_documents(self):
        """
        Load documents from the specified directory.
        Supports .txt, .pdf, and .docx file formats.
        Uses batch processing to handle large numbers of documents.
        """

        #TODO: Implement batch-processing if the number of documents is large.
        self.documents = []
        if not os.path.exists(self.document_dir):
            raise DocumentLoadError(f"Document directory not found: {self.document_dir}")

        for filename in os.listdir(self.document_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.document_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        chunks = self.chunk_document(content)
                        self.documents.extend(chunks)
                except Exception as e:
                    self.logger.warning(f"Error reading file {filename}: {str(e)}")
        
        if not self.documents:
            self.logger.warning("No documents were successfully loaded.")
        else:
            self.logger.info(f"Loaded {len(self.documents)} documents.")

    def chunk_document(self, text, max_words=400, max_sentences=15):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        word_count = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            if word_count + len(sentence_words) > max_words or len(current_chunk) >= max_sentences:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                word_count = 0
            current_chunk.append(sentence)
            word_count += len(sentence_words)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def get_chunk_info(self):
        return {
            "total_chunks": len(self.documents),
            "avg_chunk_length": sum(len(chunk.split()) for chunk in self.documents) / len(self.documents)
        }
    
    def process_batch():
        """
        Process a batch of documents.
        Args:
            batch (List[str]): A list of file paths to process.
        """
        pass

    def index_documents(self):
        if not self.documents:
            self.logger.warning("No documents to index.")
            return

        self.logger.info("Starting document indexing...")
        try:
            self.document_vectors = self.vectorizer.fit_transform(self.documents)
            self.logger.info(f"Indexed {self.document_vectors.shape[0]} documents with {self.document_vectors.shape[1]} features.")
        except Exception as e:
            self.logger.error(f"Error during document indexing: {str(e)}")

    def read_file(self, file_path):
        """
        Read the content of a file based on its extension.
        Args:
            file_path (str): The path to the file to be read.
        Returns:
            str: The content of the file, or None if the file type is unsupported.
        """
        # ... (To be implemented)
    
    def read_pdf(self, file_path):
        """
        Read the content of a PDF file.
        Args:
            file_path (str): The path to the PDF file.
        Returns:
            str: The extracted text content of the PDF.
        """
        # ... (To be implemented)
        
    def read_docx(self, file_path):
        """
        Read the content of a DOCX file.
        Args:
            file_path (str): The path to the DOCX file.
        Returns:
            str: The extracted text content of the DOCX file.
        """
        # ... (To be implemented)
    

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Process a user message through the RAG pipeline.
        Args:
            user_message (str): The user's input message.
            model_id (str): The ID of the language model to use.
            messages (List[dict]): The conversation history.
            body (dict): Additional request body parameters.
        Returns:
            Union[str, Generator, Iterator]: The enhanced prompt or response.
        """

        try:
            relevant_docs = self.get_relevant_documents(user_message, top_k=10)  # Get more docs initially
            
            context = ""
            remaining_tokens = self.max_tokens - self.count_tokens(user_message) - 100  # Reserve tokens for message and some buffer
            
            for doc in relevant_docs:
                doc_tokens = self.count_tokens(doc)
                if doc_tokens < remaining_tokens:
                    context += doc + "\n\n"
                    remaining_tokens -= doc_tokens + 2  # +2 for newlines
                else:
                    truncated_doc = self.truncate_to_token_limit(doc, remaining_tokens)
                    context += truncated_doc
                    break
            
            context = context.strip()
            enhanced_prompt = f"Context:\n{context}\n\nUser message: {user_message}"
            
            return enhanced_prompt
        except Exception as e:
            self.logger.error(f"Error in pipe method: {str(e)}")
            return "I'm sorry, but I encountered an error while processing your request."

    def index_documents(self):
        """
        Index the loaded documents using TF-IDF vectorization.
        """
        self.document_vectors = self.vectorizer.fit_transform(self.documents)
        self.logger.info(f"Indexed {len(self.documents)} documents.")
        
    def get_relevant_documents(self, query: str, top_k: int = 3) -> List[str]:
        """
        Search for relevant documents based on the given query.
        Args:
            query (str): The search query.
            top_k (int): The number of top relevant documents to return.
        Returns:
            List[str]: A list of the top_k most relevant documents.
        """

        if not query.strip():  # empty queries
            return []
        
        if self.document_vectors is None:
            self.logger.error("Documents have not been indexed. Cannot perform search.")
            return []

        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
            
            threshold = 0.1
            relevant_indices = np.where(similarities > threshold)[0]
                
            if len(relevant_indices) == 0:
                return []
                                            
            # Return top_k documents
            sorted_indices = relevant_indices[np.argsort(similarities[relevant_indices])[::-1]]

            top_indices = sorted_indices[:min(top_k, len(sorted_indices))]
            
            self.logger.info(f"Returning {len(top_indices)} documents")
        
            return [self.documents[i] for i in top_indices]
        
        except Exception as e:
            self.logger.error(f"Error during document search: {str(e)}")
            return []




            
if __name__ == "__main__":
    pipeline = Pipeline()

    pipeline.initialize()
    
    
    # Test search
    query = "example query"
    results = pipeline.get_relevant_documents(query)
    print(f"Top 5 relevant documents for '{query}':")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc[:100]}...")  # Print first 100 characters of each document

    # Test pipe method
    response = pipeline.pipe("Tell me about machine learning", "gpt-3.5-turbo", [], {})
    print("\nPipe method response:")
    print(response)