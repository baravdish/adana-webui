import os
import sys
import pytest
from unittest.mock import patch, MagicMock

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from typing import List, Union, Dict, Tuple, Callable

from backend.apps.rag.utils import get_rag_content, get_rag_context, enhance_prompt_with_rag, get_available_rag_sources
from backend.apps.rag.knowledgebank import KnowledgeBase, MyDocument, DocumentStore, CompositeKnowledgeBase

from backend.config import DOCS_TEXT_DIR
from langchain_core.documents import Document


STR_TEST_FILE_NAME = "test_file.txt"

class MockKnowledgeBase(KnowledgeBase):
    def __init__(self, documents: List[Document]):
        self.documents = documents

    def retrieve(self, query: str, k: int, embedding_function: Callable) -> List[Document]:
        return self.documents[:k]


# Mock ChromaDB client
@pytest.fixture
def mock_chroma_client():
    with patch('backend.apps.rag.utils.CHROMA_CLIENT') as mock_client:
        yield mock_client

# Test get_rag_content
@pytest.mark.asyncio
async def test_get_rag_content_file(tmp_path):
    file_content = "This is a test file content.\n"
    # file_path = tmp_path / "short_text.txt"
    file_path = tmp_path / STR_TEST_FILE_NAME
    
    file_path.write_text(file_content)
    print(f"Original DOCS_TEXT_DIR: {DOCS_TEXT_DIR}")
    with patch('backend.config.DOCS_TEXT_DIR', str(tmp_path)):
        content = await get_rag_content("file", STR_TEST_FILE_NAME)
        assert content == file_content


@pytest.mark.asyncio
async def test_get_rag_content_collection(mock_chroma_client):
    mock_collection = MagicMock()
    mock_collection.get.return_value = {"documents": [["Test collection content"]]}
    mock_chroma_client.get_collection.return_value = mock_collection
    
    content = await get_rag_content("collection", "test_collection")
    assert content == "Test collection content"

@pytest.mark.asyncio
async def test_get_rag_content_invalid_type():
    with pytest.raises(ValueError):
        await get_rag_content("invalid_type", "test")


# # Test get_rag_context
# def test_get_rag_context_old():
#     messages = [{"role": "user", "content": "Tell me about #file:" + STR_TEST_FILE_NAME}]
    
#     with patch('backend.apps.rag.utils.get_rag_content') as mock_get_content:
#         mock_get_content.return_value = "Test file content"
#         l_d_files = [
#             {"type": "text", "content": "Test file content", "collection_name": "sample1"},
#             {"type": "collection", "collection_names": ["collection1", "collection2"]}]
            
#         context, citations = get_rag_context(l_d_files, messages, None, 5, None, 0.5, False)
#         print(f"Messages: {messages}")
#         print(f"Returned context: '{context}'")
#         print(f"Returned citations: {citations}")
#         print(f"Mock called: {mock_get_content.called}")
#         if mock_get_content.called:
#             print(f"Mock call args: {mock_get_content.call_args}")

#         assert context == "Test file content"
#         assert mock_get_content.called
#         mock_get_content.assert_called_once_with("file", STR_TEST_FILE_NAME)
        
#         # assert context == "Test file content"
#         # assert citations == [{"source": "file:" + STR_TEST_FILE_NAME, "content": "Test file content.\n"}]

# Test enhance_prompt_with_rag
def test_enhance_prompt_with_rag():
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Tell me more about it."}
    ]
    rag_content = "Paris is the capital and most populous city of France."
    
    enhanced_messages = enhance_prompt_with_rag(messages, rag_content)
    
    assert len(enhanced_messages) == 4
    assert enhanced_messages[2]["role"] == "system"
    assert "Paris is the capital" in enhanced_messages[2]["content"]

# Test get_available_rag_sources
def test_get_available_rag_sources(tmp_path, mock_chroma_client):
    # Create a test file
    test_file = tmp_path / "short_text.txt"
    test_file.touch()
    
    # Mock ChromaDB collections
    mock_collection = MagicMock(name="test_collection")
    mock_collection.name = "test_collection"
    mock_chroma_client.list_collections.return_value = [mock_collection]
    
    
    def get_available_rag_sources_wrapper():
        with patch('os.listdir', return_value=[test_file.name]):
            return get_available_rag_sources()
        
    
    # NOTE: Here all sources are available
    with patch('backend.config.DOCS_TEXT_DIR', str(tmp_path)):
        # NOTE: Here only the the mock_chroma_client database source files are available
        with patch('backend.config.CHROMA_CLIENT', mock_chroma_client):
            l_d_sources = get_available_rag_sources_wrapper() # all files in the mock_chroma_client database


        # l_d_sources = get_available_rag_sources() # all files in DOCS_TEXT_DIR
        
        # TODO: Print the variable l_d_sources which is a list of dictionaries
        print(f"Available RAG sources: {l_d_sources}")
        print(f"Available RAG sources LENGTH: {len(l_d_sources)}")
        
    assert len(l_d_sources) == 2, f"Expected 2 sources, but found {len(l_d_sources)}"
    assert any(source['collection_name'] == 'short_text.txt' and source['type'] == 'file' for source in l_d_sources), "Test file not found in sources"
    assert any('collection_names' in source and 'test_collection' in source['collection_names'] and source['type'] == 'collection' for source in l_d_sources), "Test collection not found in sources"
# @pytest.mark.asyncio
# async def test_get_rag_context():
# def test_get_rag_context():
#     l_d_messages = [{"role": "user", "content": f"Tell me about #file:{STR_TEST_FILE_NAME}"}]
#     l_d_files = [
#         {"type": "text", 
#          "content": "Test file content", 
#          "collection_name": STR_TEST_FILE_NAME},
#         {"type": "collection", 
#          "collection_names": ["collection1", "collection2"]}
#     ]

#     # Mock the query_collection function instead of get_rag_content
#     with patch('backend.apps.rag.utils.query_collection') as mock_query_collection:
#         mock_query_collection.return_value = {
#             "documents": [["Collection content"]],
#             "metadatas": [{"source": "collection1"}]
#         }

#         str_context, l_d_citations = get_rag_context(l_d_files, l_d_messages, None, 5, None, 0.5, False)
#         # for citation in l_d_citations:
#         #     print(f"Citation: {citation}")
            
            
#         print(f"Messages: {l_d_messages}")
#         print(f"Returned context: '{str_context}'")
#         print(f"Returned citations: {l_d_citations}")
#         print(f"Mock called: {mock_query_collection.called}")
#         if mock_query_collection.called:
#             print(f"Mock call args: {mock_query_collection.call_args}")

#         # Check that the context includes both the text file content and the collection content
#         assert "Test file content" in str_context
#         assert "Collection content" in str_context

#         # Check that we have citations for both the text file and the collection
#         assert len(l_d_citations) == 2
#         assert any(citation['source']['collection_name'] == STR_TEST_FILE_NAME for citation in l_d_citations)
#         assert any(citation['source']['type'] == 'collection' for citation in l_d_citations)

#         # Verify that query_collection was called for the collection
#         assert mock_query_collection.called
#         mock_query_collection.assert_called_once_with(
#             collection_names=['collection1', 'collection2'],
#             query="Tell me about #file:test_file.txt",
#             embedding_function=None,
#             k=5
#         )


def test_get_rag_context():
    doc1 = MyDocument(page_content="This is document 1. Tonight I will go out and jog.",
                      metadata={"source": "source1"},
                      id="1",
                      relevance_score=0.8)
    doc2 = MyDocument(page_content="This is document 2. This morning I had a cup of tea.",
                      metadata={"source": "source2"},
                      id="2",
                      relevance_score=0.6)
    doc3 = MyDocument(page_content="This is document 3. Tomorrow at the restaurant I will have dinner at 18.00 o'clock.",
                      metadata={"source": "source3"},
                      id="3",
                      relevance_score=0.7)
    doc4 = MyDocument(page_content="This is document 4. My dog's name is Pluto.",
                      metadata={"source": "source4"},
                      id="4",
                      relevance_score=0.5)
    doc5 = MyDocument(page_content="This is document 5. My dog Pluto has a surgery at hospital at 21.00 o'clock tomorrow.",
                      metadata={"source": "source5"},
                      id="5",
                      relevance_score=0.9)
    
    doc_store = DocumentStore([doc1, doc2, doc3, doc4, doc5])
    
    mock_kb = MockKnowledgeBase(doc_store)
    mock_embedding_func = MagicMock()
    
    # Define a mock reranking function that reverses the order
    def mock_rerank(query, docs):
        return list(reversed(docs))
    
    mock_reranking_func = MagicMock(side_effect=mock_rerank)

    # Test case 1: Basic retrieval
    context, citations = get_rag_context("test query", mock_kb, mock_embedding_func, k=2)
    assert context == "This is document 1. Tonight I will go out and jog.\n\nThis is document 2. This morning I had a cup of tea."
    assert len(citations) == 2
    assert citations[0]["source"] == "source1"
    assert citations[1]["source"] == "source2"

    # Test case 2: With reranking
    context, citations = get_rag_context("test query", mock_kb, mock_embedding_func, k=2, reranking_function=mock_reranking_func)
    assert context == "This is document 5. My dog Pluto has a surgery at hospital at 21.00 o'clock tomorrow.\n\nThis is document 4. My dog's name is Pluto."
    assert citations[0]["source"] == "source5"
    assert citations[1]["source"] == "source4"
    # Test case 3: With relevance threshold
    context, citations = get_rag_context("test query", mock_kb, mock_embedding_func, k=3, relevance_threshold=0.75)
    assert len(citations) == 2  # Only two documents above the threshold

    # Test case 4: CompositeKnowledgeBase
    mock_kb1 = MockKnowledgeBase([Document("KB1 Content", "KB1", 0.95)])
    mock_kb2 = MockKnowledgeBase([Document("KB2 Content", "KB2", 0.85)])
    composite_kb = CompositeKnowledgeBase([mock_kb1, mock_kb2])
    
    context, citations = get_rag_context("test query", composite_kb, mock_embedding_func, k=2)
    assert "KB1 Content" in context and "KB2 Content" in context
    assert len(citations) == 2

    # Test case 5: Empty result
    empty_kb = MockKnowledgeBase([])
    context, citations = get_rag_context("test query", empty_kb, mock_embedding_func, k=2)
    assert context == ""
    assert len(citations) == 0

    print("All test cases passed successfully!")

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])