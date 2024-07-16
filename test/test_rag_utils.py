import os
import sys
import pytest
from unittest.mock import patch, MagicMock

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from backend.apps.rag.utils import get_rag_content, get_rag_context, enhance_prompt_with_rag, get_available_rag_sources

from backend.config import DOCS_DIR


# Mock ChromaDB client
@pytest.fixture
def mock_chroma_client():
    with patch('backend.apps.rag.utils.CHROMA_CLIENT') as mock_client:
        yield mock_client

# Test get_rag_content
@pytest.mark.asyncio
async def test_get_rag_content_file(tmp_path):
    file_content = "This is a test file content."
    file_path = tmp_path / "short_text.txt"
    file_path.write_text(file_content)
    
    with patch('backend.config.DOCS_DIR', str(tmp_path)):
        content = await get_rag_content("file", "short_text.txt")
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


# Test get_rag_context
def test_get_rag_context():
    messages = [{"role": "user", "content": "Tell me about #file:short_text.txt"}]
    
    with patch('backend.apps.rag.utils.get_rag_content') as mock_get_content:
        mock_get_content.return_value = "Test file content"
        context, citations = get_rag_context([], messages, None, 5, None, 0.5, False)
        
        assert context == "Test file content"
        assert citations == [{"source": "file:short_text.txt", "content": "Test file content"}]

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
    
    with patch('backend.config.DOCS_DIR', str(tmp_path)):
        sources = get_available_rag_sources()
        
        assert len(sources) == 2
        assert {"type": "file", "collection_name": "test_file.txt"} in sources
        assert {"type": "collection", "collection_names": ["test_collection"]} in sources

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])