import os
import sys
import pytest
from unittest.mock import patch, MagicMock

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from typing import List, Union, Dict, Tuple, Callable, Document

from backend.apps.rag.utils import get_rag_content, get_rag_context, enhance_prompt_with_rag, get_available_rag_sources
from backend.apps.rag.knowledgebank import get_rag_context, KnowledgeBase, Document, CompositeKnowledgeBase

from backend.config import DOCS_TEXT_DIR

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
    (tmp_path / "short_text.txt").touch()
    
    # Mock ChromaDB collections
    mock_chroma_client.list_collections.return_value = [MagicMock(name="test_collection")]
    
    with patch('backend.config.DOCS_TEXT_DIR', str(tmp_path)):
        l_d_sources = get_available_rag_sources()
        
        assert len(l_d_sources) == 2
        assert {"type": "file", "collection_name": STR_TEST_FILE_NAME} in l_d_sources
        assert {"type": "collection", "collection_names": ["test_collection"]} in l_d_sources


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
    # Setup
    mock_docs = [
        Document("Content 1", "Source 1", 0.9),
        Document("Content 2", "Source 2", 0.8),
        Document("Content 3", "Source 3", 0.7),
    ]
    mock_kb = MockKnowledgeBase(mock_docs)
    mock_embedding_func = MagicMock()
    mock_reranking_func = MagicMock(return_value=mock_docs[::-1])  # Reverse the order

    # Test case 1: Basic retrieval
    context, citations = get_rag_context("test query", mock_kb, mock_embedding_func, k=2)
    assert context == "Content 1\n\nContent 2"
    assert len(citations) == 2
    assert citations[0]["source"] == "Source 1"
    assert citations[1]["source"] == "Source 2"
    
    # Test case 2: With reranking
    context, citations = get_rag_context("test query", mock_kb, mock_embedding_func, k=2, reranking_function=mock_reranking_func)
    assert context == "Content 3\n\nContent 2"
    assert citations[0]["source"] == "Source 3"

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