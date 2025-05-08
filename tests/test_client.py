"""
Tests for the client module that handles Aerospike Vector Search client operations.
"""
import pytest
from unittest.mock import Mock, patch

import aerospike_vector_search as avs

from bench.client import (
    get_client,
    get_index_name,
    index_exists,
    _create_index,
    create_index
)


@pytest.fixture
def mock_avs_client():
    """Create a mock AVS client for testing."""
    client = Mock(spec=avs.Client)
    client.index_list.return_value = ["test-dataset-index", "test-dataset-index-123"]
    return client


class TestGetClient:
    """Test suite for get_client function."""

    @patch('aerospike_vector_search.Client')
    def test_get_client_with_load_balancer(self, mock_client_class):
        """Test creating a client with load balancer enabled."""
        # Arrange
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        # Act
        client = get_client("test-host", 3000, True)

        # Assert
        mock_client_class.assert_called_once()
        args, kwargs = mock_client_class.call_args
        assert kwargs['seeds'].host == "test-host"
        assert kwargs['seeds'].port == 3000
        assert kwargs['is_loadbalancer'] is True
        assert client is mock_client_instance

    @patch('aerospike_vector_search.Client')
    def test_get_client_without_load_balancer(self, mock_client_class):
        """Test creating a client with load balancer disabled."""
        # Arrange
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        # Act
        client = get_client("test-host", 3000, False)

        # Assert
        mock_client_class.assert_called_once()
        args, kwargs = mock_client_class.call_args
        assert kwargs['seeds'].host == "test-host"
        assert kwargs['seeds'].port == 3000
        assert kwargs['is_loadbalancer'] is False
        assert client is mock_client_instance


class TestGetIndexName:
    """Test suite for get_index_name function."""

    def test_get_index_name_with_id(self):
        """Test getting an index name with an ID."""
        # Act
        index_name = get_index_name("test-dataset", "123")

        # Assert
        assert index_name == "test-dataset-index-123"

    def test_get_index_name_without_id(self):
        """Test getting an index name without an ID."""
        # Act
        index_name = get_index_name("test-dataset", -1)

        # Assert
        assert index_name == "test-dataset-index"


class TestIndexExists:
    """Test suite for index_exists function."""

    def test_index_exists_when_exists(self, mock_avs_client):
        """Test checking if an index exists when it does."""
        # Act
        result = index_exists(mock_avs_client, "test-dataset-index")

        # Assert
        assert result is True
        mock_avs_client.index_list.assert_called_once()

    def test_index_exists_when_does_not_exist(self, mock_avs_client):
        """Test checking if an index exists when it doesn't."""
        # Act
        result = index_exists(mock_avs_client, "nonexistent-index")

        # Assert
        assert result is False
        mock_avs_client.index_list.assert_called_once()


class TestCreateIndex:
    """Test suite for _create_index and create_index functions."""

    def test_create_index_internal(self, mock_avs_client):
        """Test the internal _create_index function."""
        # Arrange
        namespace = "test-namespace"
        vector_field = "vector"
        index_name = "test-dataset-index"
        dimensions = 128
        distance_metric = avs.VectorDistanceMetric.SQUARED_EUCLIDEAN

        # Act
        _create_index(
            mock_avs_client,
            namespace,
            vector_field,
            index_name,
            dimensions,
            distance_metric
        )

        # Assert
        mock_avs_client.index_create.assert_called_once_with(
            namespace=namespace,
            name=index_name,
            vector_field=vector_field,
            dimensions=dimensions,
            vector_distance_metric=distance_metric
        )

    @patch('bench.client.index_exists')
    @patch('bench.client._create_index')
    def test_create_index_when_not_exists(
        self, mock_create_index, mock_index_exists, mock_avs_client
    ):
        """Test creating an index when it doesn't exist."""
        # Arrange
        index_name = "test-dataset-index-123"
        namespace = "test-namespace"
        vector_field = "vector"
        dimensions = 128
        distance_metric = avs.VectorDistanceMetric.SQUARED_EUCLIDEAN

        mock_index_exists.return_value = False

        # Act
        create_index(
            mock_avs_client,
            index_name,
            namespace,
            vector_field,
            dimensions,
            distance_metric
        )

        # Assert
        mock_index_exists.assert_called_once_with(mock_avs_client, "test-dataset-index-123")
        mock_create_index.assert_called_once_with(
            mock_avs_client,
            namespace,
            vector_field,
            "test-dataset-index-123",
            dimensions,
            distance_metric
        )

    @patch('bench.client.index_exists')
    @patch('bench.client._create_index')
    def test_create_index_when_already_exists(
        self, mock_create_index, mock_index_exists, mock_avs_client
    ):
        """Test creating an index when it already exists."""
        # Arrange
        namespace = "test-namespace"
        vector_field = "vector"
        index_name = "test-dataset-index-123"
        dimensions = 128
        distance_metric = avs.VectorDistanceMetric.SQUARED_EUCLIDEAN

        mock_index_exists.return_value = True

        # Act
        create_index(
            mock_avs_client,
            index_name,
            namespace,
            vector_field,
            dimensions,
            distance_metric
        )

        # Assert
        mock_index_exists.assert_called_once_with(mock_avs_client, "test-dataset-index-123")
        mock_create_index.assert_not_called()


class TestClientIntegration:
    """Integration tests for client functions.

    These tests require a running AVS server and are marked to be skipped by default.
    To run these tests, use: pytest -m integration
    """

    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires a running AVS server")
    def test_get_client_integration(self):
        """Test creating a real client and connecting to a server."""
        # Arrange & Act
        client = get_client("localhost", 5000, False)

        # Assert
        assert client is not None
        # Verify connection by performing a simple operation
        try:
            # Try a simple operation like fetching cluster info
            info = client.info()
            assert info is not None
        finally:
            # Clean up
            client.close()

    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires a running AVS server")
    def test_create_index_integration(self):
        """Test creating a real index on a server."""
        # Arrange
        client = get_client("localhost", 5000, False)
        dataset_name = "test-dataset-integration"
        namespace = "test"
        vector_field = "vector"
        index_name = get_index_name(dataset_name, "integration-test")
        dimensions = 4
        distance_metric = avs.VectorDistanceMetric.SQUARED_EUCLIDEAN

        try:
            # Act
            create_index(
                client,
                index_name,
                namespace,
                vector_field,
                dimensions,
                distance_metric
            )

            # Assert
            assert index_exists(client, index_name)

        finally:
            # Clean up - try to delete the index if possible
            try:
                client.index_remove(namespace, index_name)
            except Exception:
                pass
            client.close()
