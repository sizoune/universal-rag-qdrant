import pytest
from unittest.mock import patch, MagicMock
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from src.vector_store import initialize_vector_store
from src.config import config


@patch("src.vector_store.get_qdrant_client")
@patch("src.vector_store.get_embedder")
def test_strict_dimension_checking_fails(mock_get_embedder, mock_get_client):
    """
    Test that ValueError is raised when existing collection dimension
    does not match the .env EMBEDDER_DIMENSION.
    """
    mock_client = MagicMock(spec=QdrantClient)
    mock_get_client.return_value = mock_client

    # Mock Embedder to just be a dummy
    mock_get_embedder.return_value = MagicMock()

    # Mock collection info returned by Qdrant with dimension 4096 (while .env expects 1536 from conftest)
    mock_collection_info = MagicMock()
    mock_collection_info.config.params.vectors.size = 4096
    mock_client.get_collection.return_value = mock_collection_info

    with pytest.raises(ValueError, match="Strict Dimension Checking Failed"):
        initialize_vector_store()


@patch("src.vector_store.get_qdrant_client")
@patch("src.vector_store.get_embedder")
def test_strict_dimension_checking_passes(mock_get_embedder, mock_get_client):
    """
    Test that it passes successfully when dimensions match.
    """
    mock_client = MagicMock(spec=QdrantClient)
    mock_get_client.return_value = mock_client
    mock_get_embedder.return_value = MagicMock()

    # Mock matching dimension
    mock_collection_info = MagicMock()
    mock_collection_info.config.params.vectors.size = config.EMBEDDER_DIMENSION
    mock_client.get_collection.return_value = mock_collection_info

    vector_store = initialize_vector_store()

    # Should not raise exception and should call get_collection
    mock_client.get_collection.assert_called_once_with(config.QDRANT_COLLECTION_NAME)
    assert vector_store is not None


@patch("src.vector_store.get_qdrant_client")
@patch("src.vector_store.get_embedder")
def test_collection_creation_on_not_found(mock_get_embedder, mock_get_client):
    """
    Test that a new collection is created if one does not exist.
    """
    mock_client = MagicMock(spec=QdrantClient)
    mock_get_client.return_value = mock_client
    mock_get_embedder.return_value = MagicMock()

    # Simulate collection not found
    mock_client.get_collection.side_effect = Exception("Not found: Collection")

    initialize_vector_store()

    # Verify create_collection was called
    mock_client.create_collection.assert_called_once()
    args, kwargs = mock_client.create_collection.call_args
    assert kwargs["collection_name"] == config.QDRANT_COLLECTION_NAME
    assert kwargs["vectors_config"].size == config.EMBEDDER_DIMENSION
