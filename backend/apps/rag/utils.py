import os
import logging
import requests

from typing import List, Union, Dict, Tuple, Callable, Any
import re

from apps.ollama.main import (
    generate_ollama_embeddings,
    GenerateEmbeddingsForm,
)

from huggingface_hub import snapshot_download

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)

from typing import Optional

from utils.misc import get_last_user_message, add_or_update_system_message
from config import SRC_LOG_LEVELS, CHROMA_CLIENT

# =============== Adana ===============
# from fastapi import HTTPException
from apps.webui.models.documents import Documents
from backend.config import DOCS_TEXT_DIR
# import backend.config
from functools import lru_cache
from backend.apps.rag.knowledgebank import KnowledgeBase, DocumentStore, Document, MyDocument
# In backend/apps/rag/utils.py
from backend.apps.rag.reranking import ReRanker


reranker = ReRanker()

# =============== Adana ===============

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])

@lru_cache(maxsize=100)
def get_rag_content_cached(rag_type: str, rag_name: str) -> str:
    return get_rag_content(rag_type, rag_name)

def get_available_rag_sources() -> List[Dict[str, Union[str, List[str]]]]:
    """
    Get a list of available RAG sources including local files and ChromaDB collections.
    
    Returns:
        List[Dict[str, Union[str, List[str]]]]: A list of dictionaries, each representing a RAG source.
        For files: {"type": "file", "collection_name": filename}
        For collections: {"type": "collection", "collection_names": [collection_name]}
    """
    l_d_rag_sources = []
            
    if os.path.exists(DOCS_TEXT_DIR):
        for filename in os.listdir(DOCS_TEXT_DIR):
            if os.path.isfile(os.path.join(DOCS_TEXT_DIR, filename)):
                print(f"Adding file source: {filename}")
                l_d_rag_sources.append({
                    "type": "file",
                    "collection_name": filename
                })
    else:
        log.warning(f"========== Directory '{DOCS_TEXT_DIR}' does not exist ==========")

    print("Checking ChromaDB collections")
    # Get ChromaDB collections
    try:
        collections = CHROMA_CLIENT.list_collections()
        for collection in collections:
            print(f"Adding collection source: {collection.name}")
            l_d_rag_sources.append({
                "type": "collection",
                "collection_names": [collection.name]
            })
    except Exception as e:
        log.error(f"Error fetching ChromaDB collections: {e}")

    
    return l_d_rag_sources

async def get_rag_content(rag_type: str, rag_name: str) -> str:
    """
    Retrieve content based on the RAG reference type and name.
    """
    if rag_type == "file":
        print("rag_name", rag_name)
        print("XXXXXXXXXXXXXXXXXXXXXXx DOCS_TEXT_DIR XXXXXXXXXXXXXXXXXXXx´ ", DOCS_TEXT_DIR)
        file_path = os.path.join(DOCS_TEXT_DIR, rag_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{rag_name}' not found")
        with open(file_path, 'r') as file:
            return file.read()
    elif rag_type == "collection":
        try:
            collection = CHROMA_CLIENT.get_collection(name=rag_name)
            d_results = collection.get()
            if d_results['documents']:
                return "\n\n".join(d_results['documents'][0])
            else:
                raise FileNotFoundError(f"Collection '{rag_name}' is empty")
        except Exception as e:
            raise FileNotFoundError(f"Collection '{rag_name}' not found or error occurred: {str(e)}")
    else:
        raise ValueError(f"Invalid RAG type: {rag_type}")
    
    
# def enhance_prompt_with_rag(messages: List[Dict[str, str]], rag_content: str) -> List[Dict[str, str]]:
def enhance_prompt_with_rag(messages: List[dict], rag_content: str) -> List[dict]:
    """
    Enhance the chat messages with RAG content.
    """
    rag_message = {
        "role": "system",
        "content": f"Here is some relevant information:\n\n{rag_content}\n\nPlease use this information to inform your response to the user's query."
    }
    
    # Insert the RAG message before the last user message
    insert_index = len(messages) - 1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            insert_index = i
            break
    
    messages.insert(insert_index, rag_message)
    return messages

def get_rag_context(
    files,
    messages,
    embedding_function,
    k,
    reranking_function,
    r,
    hybrid_search,
):
    log.debug(f"files: {files} {messages} {embedding_function} {reranking_function}")
    query = get_last_user_message(messages)

    # Check for RAG references in the query
    rag_reference_match = re.search(r'#(file|collection):\s*(\S+)', query)
    if rag_reference_match:
        rag_type, rag_name = rag_reference_match.groups()
        try:
            rag_content = get_rag_content_cached(rag_type, rag_name)
            # Remove the RAG reference from the query
            query = re.sub(r'#(file|collection):\s*(\S+)', '', query).strip()
            return rag_content, [{"source": f"{rag_type}:{rag_name}", "content": rag_content}]
        except FileNotFoundError as e:
            log.warning(f"RAG content not found: {str(e)}")
            # Continue with normal search if RAG content is not found

    # Existing code for context retrieval
    extracted_collections = []
    relevant_contexts = []

    for file in files:
        context = None

        collection_names = (
            file["collection_names"]
            if file["type"] == "collection"
            else [file["collection_name"]]
        )

        collection_names = set(collection_names).difference(extracted_collections)
        if not collection_names:
            log.debug(f"skipping {file} as it has already been extracted")
            continue

        try:
            if file["type"] == "text":
                context = file["content"]
            else:
                if hybrid_search:
                    context = query_collection_with_hybrid_search(
                        collection_names=collection_names,
                        query=query,
                        embedding_function=embedding_function,
                        k=k,
                        reranking_function=reranking_function,
                        r=r,
                    )
                else:
                    context = query_collection(
                        collection_names=collection_names,
                        query=query,
                        embedding_function=embedding_function,
                        k=k,
                    )
        except Exception as e:
            log.exception(e)
            context = None

        if context:
            relevant_contexts.append({**context, "source": file})

        extracted_collections.extend(collection_names)

    contexts = []
    citations = []

    for context in relevant_contexts:
        try:
            if "documents" in context:
                contexts.append(
                    "\n\n".join(
                        [text for text in context["documents"][0] if text is not None]
                    )
                )

                if "metadatas" in context:
                    citations.append(
                        {
                            "source": context["source"],
                            "document": context["documents"][0],
                            "metadata": context["metadatas"][0],
                        }
                    )
        except Exception as e:
            log.exception(e)

    return "\n\n".join(contexts), citations

def query_doc(
    collection_name: str,
    query: str,
    embedding_function,
    k: int,
):
    try:
        collection = CHROMA_CLIENT.get_collection(name=collection_name)
        query_embeddings = embedding_function(query)

        result = collection.query(
            query_embeddings=[query_embeddings],
            n_results=k,
        )

        log.info(f"query_doc:result {result}")
        return result
    except Exception as e:
        raise e


def query_doc_with_hybrid_search(
    collection_name: str,
    query: str,
    embedding_function,
    k: int,
    reranking_function,
    r: float,
):
    try:
        collection = CHROMA_CLIENT.get_collection(name=collection_name)
        documents = collection.get()  # get all documents

        bm25_retriever = BM25Retriever.from_texts(
            texts=documents.get("documents"),
            metadatas=documents.get("metadatas"),
        )
        bm25_retriever.k = k

        chroma_retriever = ChromaRetriever(
            collection=collection,
            embedding_function=embedding_function,
            top_n=k,
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
        )

        compressor = RerankCompressor(
            embedding_function=embedding_function,
            top_n=k,
            reranking_function=reranking_function,
            r_score=r,
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble_retriever
        )

        result = compression_retriever.invoke(query)
        result = {
            "distances": [[d.metadata.get("score") for d in result]],
            "documents": [[d.page_content for d in result]],
            "metadatas": [[d.metadata for d in result]],
        }

        log.info(f"query_doc_with_hybrid_search:result {result}")
        return result
    except Exception as e:
        raise e


def merge_and_sort_query_results(query_results, k, reverse=False):
    # Initialize lists to store combined data
    combined_distances = []
    combined_documents = []
    combined_metadatas = []

    for data in query_results:
        combined_distances.extend(data["distances"][0])
        combined_documents.extend(data["documents"][0])
        combined_metadatas.extend(data["metadatas"][0])

    # Create a list of tuples (distance, document, metadata)
    combined = list(zip(combined_distances, combined_documents, combined_metadatas))

    # Sort the list based on distances
    combined.sort(key=lambda x: x[0], reverse=reverse)

    # We don't have anything :-(
    if not combined:
        sorted_distances = []
        sorted_documents = []
        sorted_metadatas = []
    else:
        # Unzip the sorted list
        sorted_distances, sorted_documents, sorted_metadatas = zip(*combined)

        # Slicing the lists to include only k elements
        sorted_distances = list(sorted_distances)[:k]
        sorted_documents = list(sorted_documents)[:k]
        sorted_metadatas = list(sorted_metadatas)[:k]

    # Create the output dictionary
    result = {
        "distances": [sorted_distances],
        "documents": [sorted_documents],
        "metadatas": [sorted_metadatas],
    }

    return result


def query_collection(
    collection_names: List[str],
    query: str,
    embedding_function,
    k: int,
):
    results = []
    for collection_name in collection_names:
        try:
            result = query_doc(
                collection_name=collection_name,
                query=query,
                k=k,
                embedding_function=embedding_function,
            )
            results.append(result)
        except:
            pass
    return merge_and_sort_query_results(results, k=k)


def query_collection_with_hybrid_search(
    collection_names: List[str],
    query: str,
    embedding_function,
    k: int,
    reranking_function,
    r: float,
):
    results = []
    for collection_name in collection_names:
        try:
            result = query_doc_with_hybrid_search(
                collection_name=collection_name,
                query=query,
                embedding_function=embedding_function,
                k=k,
                reranking_function=reranking_function,
                r=r,
            )
            results.append(result)
        except:
            pass
    return merge_and_sort_query_results(results, k=k, reverse=True)


def rag_template(template: str, context: str, query: str):
    template = template.replace("[context]", context)
    template = template.replace("[query]", query)
    return template


def get_embedding_function(
    embedding_engine,
    embedding_model,
    embedding_function,
    openai_key,
    openai_url,
    batch_size,
):
    if embedding_engine == "":
        return lambda query: embedding_function.encode(query).tolist()
    elif embedding_engine in ["ollama", "openai"]:
        if embedding_engine == "ollama":
            func = lambda query: generate_ollama_embeddings(
                GenerateEmbeddingsForm(
                    **{
                        "model": embedding_model,
                        "prompt": query,
                    }
                )
            )
        elif embedding_engine == "openai":
            func = lambda query: generate_openai_embeddings(
                model=embedding_model,
                text=query,
                key=openai_key,
                url=openai_url,
            )

        def generate_multiple(query, f):
            if isinstance(query, list):
                if embedding_engine == "openai":
                    embeddings = []
                    for i in range(0, len(query), batch_size):
                        embeddings.extend(f(query[i : i + batch_size]))
                    return embeddings
                else:
                    return [f(q) for q in query]
            else:
                return f(query)

        return lambda query: generate_multiple(query, func)

def get_rag_context(
    query: str,
    knowledge_base: KnowledgeBase,
    embedding_function: Callable,
    k: int,
    reranking_function: Optional[Callable] = None,
    relevance_threshold: float = 0.0
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retrieve relevant context for a given query from a knowledge base.

    :param query: The user's query
    :param knowledge_base: An object representing the accessible knowledge
    :param embedding_function: Function to create embeddings
    :param k: Number of top results to retrieve
    :param reranking_function: Optional function to rerank results
    :param relevance_threshold: Minimum relevance score for inclusion
    :return: Tuple of (context string, list of citations)
    """
    
    #TODO: Use doc_store objects from KnowledgeBase to retrieve documents
    # This could be called from get_rag_context()
    # relevant_docs = doc_store.retrieve("document", k=2, embedding_function=lambda x: x)
    
    
    # Retrieve relevant documents
    relevant_docs = knowledge_base.retrieve(query, k, embedding_function)

    # Rerank if a reranking function is provided
    if reranking_function:
        relevant_docs = reranking_function(query, relevant_docs)
    else:
        relevant_docs = reranker.rerank_documents(query, relevant_docs)
        
    # Filter by relevance threshold
    relevant_docs = [doc for doc in relevant_docs if doc.relevance_score >= relevance_threshold]

    # Compile context and citations
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    citations = [
        {
            "source": doc.source,
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in relevant_docs
    ]

    return context, citations


def get_rag_context_old(
    l_d_files,
    l_d_messages,
    embedding_function,
    k,
    reranking_function,
    r,
    use_hybrid_search,
):
    """
    Retrieves relevant context for a given query using RAG (Retrieval-Augmented Generation).

    Args:
        files (List[Dict[str, Any]]): A list of dictionaries representing files or collections.
            Each dictionary should have the following structure:
            - For text files: {"type": "text", "content": str, "collection_name": str}
            - For collections: {"type": "collection", "collection_names": List[str]}

        messages (List[Dict[str, str]]): A list of message dictionaries, each containing 'role' and 'content'.
            The last user message is used as the query.

        embedding_function (Callable): A function to generate embeddings for the query and documents.

        k (int): The number of top results to retrieve.

        reranking_function (Optional[Callable]): A function to rerank the retrieved results. Can be None.

        r (float): A relevance threshold for filtering results.

        hybrid_search (bool): If True, use hybrid search (combination of semantic and keyword search).
                              If False, use only semantic search.

    Returns:
        Tuple[str, List[Dict[str, Any]]]: A tuple containing:
            - A string of concatenated relevant context.
            - A list of citation dictionaries, each containing 'source', 'document', and 'metadata'.

    Note:
        This function processes the input files/collections, retrieves relevant context based on the query,
        and returns the context along with citation information.
        
    ======= DOCUMENTATION =======
    For text files:
        {
            "type": "text",
            "content": "The actual text content of the file",
            "collection_name": "unique_identifier_for_this_file"
        }
    
    For collections:
    {
        "type": "collection",
        "collection_names": ["name_of_collection1", "name_of_collection2", ...]
    }
            
    """
    
    log.debug(f"files: {l_d_files} {l_d_messages} {embedding_function} {reranking_function}")
    query = get_last_user_message(l_d_messages)

    extracted_collections = []
    l_relevant_contexts = []
    print(f"=====================utils.py files {len(l_d_files)}")
    print(f"=====================utils.py file: {l_d_files}")
    
    # {"type": "text", "content": "Test file content", "collection_name": STR_TEST_FILE_NAME},
    # {"type": "collection", "collection_names": ["collection1", "collection2"]}

    for file in l_d_files:
        d_context = None
        
        l_collection_names = (
            file["collection_names"]
            if file["type"] == "collection"
            else [file["collection_name"]]
        )

        
        for collection in l_collection_names:
            print(f"=====================utils.py collection: {collection}")
            print(f"=====================utils.py file: {file}")
            print(f"=====================utils.py file type: {file['type']}")
            if "content" in file:
                print(f"=====================utils.py file content: {file['content']}")
            if "collection_name" in file:
                print(f"=====================utils.py file collection_name: {file['collection_name']}")
            if "collection_names" in file:
                print(f"=====================utils.py file collection_names: {file['collection_names']}")
                                
        l_collection_names = set(l_collection_names).difference(extracted_collections)

        if not l_collection_names:
            log.debug(f"skipping {file} as it has already been extracted")
            continue

        try:
            if file["type"] == "text":
                d_context = {"content": file["content"]}
            else:
                if use_hybrid_search:
                    d_context = query_collection_with_hybrid_search(
                        collection_names=l_collection_names,
                        query=query,
                        embedding_function=embedding_function,
                        k=k,
                        reranking_function=reranking_function,
                        r=r,
                    )
                else:
                    d_context = query_collection(
                        collection_names=l_collection_names,
                        query=query,
                        embedding_function=embedding_function,
                        k=k,
                    )
        except Exception as e:
            log.exception(e)
            d_context = None

        if d_context:
            l_relevant_contexts.append({**d_context, "source": file})

        extracted_collections.extend(l_collection_names)



    str_context_string = ""

    l_d_citations = []
    for d_context in l_relevant_contexts:        
        
        try:
            if "documents" in d_context:
                print(f"ADDS d_context['documents'][0] {d_context['documents'][0]} TO STRING {str_context_string}")
                
                str_context_string += "\n\n".join(
                    [text for text in d_context["documents"][0] if text is not None]
                )
            elif "content" in d_context:  # Handle the text file case
                print(f"ADDS d_context['content'] {d_context['content']} TO STRING {str_context_string}")
                str_context_string += d_context["content"]
                if "metadatas" in d_context:
                    l_d_citations.append(
                        {
                            "source": d_context["source"],
                            "document": d_context["documents"][0],
                            "metadata": d_context["metadatas"][0],
                        }
                    )
            else:  # Handle the text file case
                l_d_citations.append(
                    {
                        "source": d_context["source"],
                        "document": d_context["content"],
                        "metadata": {},
                    }
                )
        except Exception as e:
            log.exception(e)
        
        for key, value in d_context.items():
            print(f"=====================utils.py d_context.key: {key}")
            print(f"=====================utils.py d_context.value: {value}")
        print(f"=====================utils.py str_context_string: {str_context_string}")
        
    str_context_string = str_context_string.strip()

    print(f"=====================ENDING utils.py str_context_string: {str_context_string}")
    print(f"=====================ENDING utils.py LENGTH l_d_citations: {len(l_d_citations)}")
    print(f"=====================ENDING utils.py l_d_citations: {l_d_citations}")
    # print(f"=====================ENDING utils.py l_d_citations[0]: {l_d_citations[0]}")
    for citation in l_d_citations:
        # Access the dictionary items using the citation variable
        # Do something with the citation dictionary
        # For example, print the values of 'source', 'document', and 'metadata'
        print("Source:", citation["source"])
        print("Document:", citation["document"])
        print("Metadata:", citation["metadata"])
        print()  # Print an empty line for separation

    
    return str_context_string, l_d_citations


def get_model_path(model: str, update_model: bool = False):
    # Construct huggingface_hub kwargs with local_files_only to return the snapshot path
    cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")

    local_files_only = not update_model

    snapshot_kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }

    log.debug(f"model: {model}")
    log.debug(f"snapshot_kwargs: {snapshot_kwargs}")

    # Inspiration from upstream sentence_transformers
    if (
        os.path.exists(model)
        or ("\\" in model or model.count("/") > 1)
        and local_files_only
    ):
        # If fully qualified path exists, return input, else set repo_id
        return model
    elif "/" not in model:
        # Set valid repo_id for model short-name
        model = "sentence-transformers" + "/" + model

    snapshot_kwargs["repo_id"] = model

    # Attempt to query the huggingface_hub library to determine the local path and/or to update
    try:
        model_repo_path = snapshot_download(**snapshot_kwargs)
        log.debug(f"model_repo_path: {model_repo_path}")
        return model_repo_path
    except Exception as e:
        log.exception(f"Cannot determine model snapshot path: {e}")
        return model


def generate_openai_embeddings(
    model: str,
    text: Union[str, list[str]],
    key: str,
    url: str = "https://api.openai.com/v1",
):
    if isinstance(text, list):
        embeddings = generate_openai_batch_embeddings(model, text, key, url)
    else:
        embeddings = generate_openai_batch_embeddings(model, [text], key, url)

    return embeddings[0] if isinstance(text, str) else embeddings


def generate_openai_batch_embeddings(
    model: str, texts: list[str], key: str, url: str = "https://api.openai.com/v1"
) -> Optional[list[list[float]]]:
    try:
        r = requests.post(
            f"{url}/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            json={"input": texts, "model": model},
        )
        r.raise_for_status()
        data = r.json()
        if "data" in data:
            return [elem["embedding"] for elem in data["data"]]
        else:
            raise "Something went wrong :/"
    except Exception as e:
        print(e)
        return None


from typing import Any

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


class ChromaRetriever(BaseRetriever):
    collection: Any
    embedding_function: Any
    top_n: int

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        query_embeddings = self.embedding_function(query)

        results = self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=self.top_n,
        )

        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]

        results = []
        for idx in range(len(ids)):
            results.append(
                Document(
                    metadata=metadatas[idx],
                    page_content=documents[idx],
                )
            )
        return results


import operator

from typing import Optional, Sequence

from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.callbacks import Callbacks
from langchain_core.pydantic_v1 import Extra

from sentence_transformers import util


class RerankCompressor(BaseDocumentCompressor):
    embedding_function: Any
    top_n: int
    reranking_function: Any
    r_score: float

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        reranking = self.reranking_function is not None

        if reranking:
            scores = self.reranking_function.predict(
                [(query, doc.page_content) for doc in documents]
            )
        else:
            query_embedding = self.embedding_function(query)
            document_embedding = self.embedding_function(
                [doc.page_content for doc in documents]
            )
            scores = util.cos_sim(query_embedding, document_embedding)[0]

        docs_with_scores = list(zip(documents, scores.tolist()))
        if self.r_score:
            docs_with_scores = [
                (d, s) for d, s in docs_with_scores if s >= self.r_score
            ]

        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        final_results = []
        for doc, doc_score in result[: self.top_n]:
            metadata = doc.metadata
            metadata["score"] = doc_score
            doc = Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
            final_results.append(doc)
        return final_results
