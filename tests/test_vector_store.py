import pytest
from unittest.mock import patch, MagicMock
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_core.embeddings import Embeddings


def _get_config():
    """Helper to get the reloaded config instance."""
    from src.config import config

    return config


def _get_initialize():
    """Helper to get the reloaded initialize_vector_store function."""
    from src.vector_store import initialize_vector_store

    return initialize_vector_store


@patch("src.vector_store.get_qdrant_client")
@patch("src.vector_store.get_embedder")
def test_strict_dimension_checking_fails(mock_get_embedder, mock_get_client):
    """
    Test that ValueError is raised when existing collection dimension
    does not match the .env EMBEDDER_DIMENSION (1536 from conftest).
    """
    config = _get_config()
    initialize_vector_store = _get_initialize()

    mock_client = MagicMock(spec=QdrantClient)
    mock_get_client.return_value = mock_client

    # Langchain Qdrant internally validates dimensions by calling embed_documents
    mock_embeddings = MagicMock(spec=Embeddings)
    mock_embeddings.embed_documents.return_value = [[0.0] * config.EMBEDDER_DIMENSION]
    mock_get_embedder.return_value = mock_embeddings

    # Mock collection with dimension 768 (mismatch with conftest's 1536)
    mock_collection_info = MagicMock()
    mock_vector_params = MagicMock()
    mock_vector_params.size = 768
    mock_vector_params.distance = rest.Distance.COSINE
    mock_collection_info.config.params.vectors = mock_vector_params
    mock_client.get_collection.return_value = mock_collection_info

    with pytest.raises(ValueError, match="Strict Dimension Checking Failed"):
        initialize_vector_store()


@patch("src.vector_store.get_qdrant_client")
@patch("src.vector_store.get_embedder")
def test_strict_dimension_checking_passes(mock_get_embedder, mock_get_client):
    """
    Test that it passes successfully when dimensions match.
    """
    config = _get_config()
    initialize_vector_store = _get_initialize()

    mock_client = MagicMock(spec=QdrantClient)
    mock_get_client.return_value = mock_client

    # Langchain Qdrant internally validates dimensions by calling embed_documents
    mock_embeddings = MagicMock(spec=Embeddings)
    mock_embeddings.embed_documents.return_value = [[0.0] * config.EMBEDDER_DIMENSION]
    mock_get_embedder.return_value = mock_embeddings

    # Mock matching dimension and distance
    mock_collection_info = MagicMock()
    mock_vector_params = MagicMock()
    mock_vector_params.size = config.EMBEDDER_DIMENSION
    mock_vector_params.distance = rest.Distance.COSINE
    mock_collection_info.config.params.vectors = mock_vector_params
    mock_client.get_collection.return_value = mock_collection_info

    vector_store = initialize_vector_store()

    # Should not raise exception and should call get_collection
    mock_client.get_collection.assert_any_call(config.QDRANT_COLLECTION_NAME)
    assert vector_store is not None


@patch("src.vector_store.get_qdrant_client")
@patch("src.vector_store.get_embedder")
def test_collection_creation_on_not_found(mock_get_embedder, mock_get_client):
    """
    Test that a new collection is created if one does not exist.
    """
    config = _get_config()
    initialize_vector_store = _get_initialize()

    mock_client = MagicMock(spec=QdrantClient)
    mock_get_client.return_value = mock_client

    # Langchain Qdrant internally validates dimensions by calling embed_documents
    mock_embeddings = MagicMock(spec=Embeddings)
    mock_embeddings.embed_documents.return_value = [[0.0] * config.EMBEDDER_DIMENSION]
    mock_get_embedder.return_value = mock_embeddings

    # Simulate collection not found initially, but found after creation
    def side_effect(*args, **kwargs):
        if not mock_client.create_collection.called:
            raise Exception("Not found: Collection")
        mock_collection_info = MagicMock()
        mock_vector_params = MagicMock()
        mock_vector_params.size = config.EMBEDDER_DIMENSION
        mock_vector_params.distance = rest.Distance.COSINE
        mock_collection_info.config.params.vectors = mock_vector_params
        return mock_collection_info

    mock_client.get_collection.side_effect = side_effect

    initialize_vector_store()

    # Verify create_collection was called
    mock_client.create_collection.assert_called_once()
    args, kwargs = mock_client.create_collection.call_args
    assert kwargs["collection_name"] == config.QDRANT_COLLECTION_NAME
    assert kwargs["vectors_config"].size == config.EMBEDDER_DIMENSION
