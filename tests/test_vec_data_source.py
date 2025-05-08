"""
Tests for the VecDataSource class that handles vector data files.
"""
import os
import pytest
import numpy as np

from bench.data_set import DataSet, VectorDataPoint, QueryDataPoint, Neighbors
from bench.vec_data_source import VecDataSource


@pytest.fixture
def test_dataset() -> DataSet:
    """Create a test dataset using the test data files."""
    base_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    # Just use fvecs for testing as we have dedicated tests for the other formats
    base_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.fvecs')]
    return DataSet(
        name="test_dataset",
        base_files=base_files,
        query_file=os.path.join(base_dir, "query.fvecs"),
        ground_truth_file=os.path.join(base_dir, "groundtruth.ivecs")
    )


@pytest.fixture
def vec_data_source(test_dataset) -> VecDataSource:
    """Create a VecDataSource instance with the test dataset."""
    return VecDataSource(test_dataset)


class TestVecDataSource:
    """Test suite for VecDataSource class."""

    def test_initialization(self, vec_data_source, test_dataset):
        """Test initialization of VecDataSource."""
        assert vec_data_source.dataset == test_dataset
        assert vec_data_source.base_vector_count is None
        assert vec_data_source._dimensions is None

    def test_is_gzipped(self, vec_data_source, tmp_path):
        """Test detection of gzipped files."""
        # Test a non-gzipped file
        base_file = os.path.join(os.path.dirname(__file__), 'test_data', 'base_0.fvecs')
        assert not vec_data_source._is_gzipped(base_file)

        # Test a gzipped file
        gzipped_file = os.path.join(os.path.dirname(__file__), 'test_data', 'base_0.fvecs.gz')
        assert vec_data_source._is_gzipped(gzipped_file)

        # Test empty file
        empty_file = tmp_path / "empty.fvecs"
        with open(empty_file, 'wb'):
            pass
        assert not vec_data_source._is_gzipped(str(empty_file))

    def test_is_bvecs_format(self, vec_data_source):
        """Test detection of bvecs format."""
        # Test a bvecs file
        bvecs_file = "test_file.bvecs"
        assert vec_data_source._is_bvecs_format(bvecs_file)

        # Test a non-bvecs file
        non_bvecs_file = "test_file.fvecs"
        assert not vec_data_source._is_bvecs_format(non_bvecs_file)

    def test_read_vectors_fvecs(self, vec_data_source):
        """Test reading vectors from an fvecs file."""
        fvecs_file = os.path.join(os.path.dirname(__file__), 'test_data', 'base_0.fvecs')
        vectors = list(vec_data_source.read_vectors(fvecs_file))

        # Check that we got the expected number of vectors
        assert len(vectors) == 10

        # Check that each vector has the expected structure
        for i, vector in enumerate(vectors):
            assert isinstance(vector, VectorDataPoint)
            assert vector.id == i
            assert vector.vector.shape == (4,)
            assert vector.vector.dtype == np.float32

    def test_read_vectors_ivecs(self, vec_data_source):
        """Test reading vectors from an ivecs file."""
        ivecs_file = os.path.join(os.path.dirname(__file__), 'test_data', 'base_0.ivecs')
        vectors = list(vec_data_source.read_vectors(ivecs_file))

        # Check that we got the expected number of vectors
        assert len(vectors) == 10

        # Check that each vector has the expected structure
        for i, vector in enumerate(vectors):
            assert isinstance(vector, VectorDataPoint)
            assert vector.id == i
            assert vector.vector.shape == (4,)
            assert vector.vector.dtype == np.float32

    def test_read_vectors_bvecs(self, vec_data_source):
        """Test reading vectors from a bvecs file."""
        bvecs_file = os.path.join(os.path.dirname(__file__), 'test_data', 'base_0.bvecs')
        vectors = list(vec_data_source.read_vectors(bvecs_file))

        # Check that we got the expected number of vectors
        assert len(vectors) == 10

        # Check that each vector has the expected structure
        for i, vector in enumerate(vectors):
            assert isinstance(vector, VectorDataPoint)
            assert vector.id == i
            assert vector.vector.shape == (4,)
            assert vector.vector.dtype == np.float32

    def test_read_vectors_nonexistent_file(self, vec_data_source):
        """Test reading vectors from a nonexistent file raises an error."""
        nonexistent_file = os.path.join(os.path.dirname(__file__), 'test_data', 'nonexistent.fvecs')
        with pytest.raises(ValueError, match="File not found"):
            list(vec_data_source.read_vectors(nonexistent_file))

    def test_read_base_vectors(self, vec_data_source):
        """Test reading base vectors."""
        # Read base vectors from file index 0
        vectors = list(vec_data_source.read_base_vectors(0))

        # Check that we got the expected number of vectors
        assert len(vectors) == 10

        # Check that each vector has the expected structure
        for i, vector in enumerate(vectors):
            assert isinstance(vector, VectorDataPoint)
            assert vector.id == i
            assert vector.vector.shape == (4,)

    def test_read_base_vectors_nonexistent_index(self, vec_data_source):
        """Test reading base vectors from a nonexistent index raises an error."""
        with pytest.raises(ValueError, match="Base file index .* out of range"):
            list(vec_data_source.read_base_vectors(999))

    def test_read_query_vectors(self, vec_data_source):
        """Test reading query vectors with their neighbors."""
        query_vectors = list(vec_data_source.read_query_vectors())

        # Check that we got the expected number of query vectors
        assert len(query_vectors) == 5

        # Check that each query vector has the expected structure
        for _, query in enumerate(query_vectors):
            assert isinstance(query, QueryDataPoint)
            assert query.vector.shape == (4,)
            assert isinstance(query.neighbors, list)
            assert len(query.neighbors) == 3  # We created test data with 3 neighbors per query

    def test_get_dimensions(self, vec_data_source):
        """Test getting dimensions of vectors."""
        dimensions = vec_data_source.get_dimensions()
        assert dimensions == 4  # Our test data has 4-dimensional vectors

        # Check that the dimensions are cached
        assert vec_data_source._dimensions == 4

    def test_get_num_base_vectors(self, vec_data_source):
        """Test getting the number of base vectors."""
        num_vectors = vec_data_source.get_num_base_vectors()
        assert num_vectors == 10  # Our test data has 10 vectors per base file

        # Check that the count is cached
        assert vec_data_source.base_vector_count == 10

    def test_get_vector_distance_metric(self, vec_data_source):
        """Test getting the vector distance metric."""
        metric = vec_data_source.get_vector_distance_metric()
        assert metric == "squared_euclidean"

    def test_read_neighbors(self, vec_data_source):
        """Test reading ground truth neighbors."""
        ground_truth_file = vec_data_source.dataset.ground_truth_file
        neighbors = vec_data_source.read_neighbors(ground_truth_file)

        # Check that we got the expected number of neighbor lists
        assert len(neighbors) == 5  # Our test data has 5 queries with neighbors

        # Check that each neighbor list has the expected structure
        for i, neighbor in enumerate(neighbors):
            assert isinstance(neighbor, Neighbors)
            assert neighbor.id == i
            assert isinstance(neighbor.neighbors, list)
            assert len(neighbor.neighbors) == 3  # Each query has 3 neighbors in our test data

    def test_nonexistent_query_file(self, test_dataset):
        """Test handling of nonexistent query file."""
        # Create a dataset with a nonexistent query file
        bad_dataset = DataSet(
            name="bad_dataset",
            base_files=test_dataset.base_files,
            query_file="nonexistent.fvecs",
            ground_truth_file=test_dataset.ground_truth_file
        )
        data_source = VecDataSource(bad_dataset)

        with pytest.raises(ValueError, match="Query file not found"):
            list(data_source.read_query_vectors())

    def test_nonexistent_ground_truth_file(self, test_dataset):
        """Test handling of nonexistent ground truth file."""
        # Create a dataset with a nonexistent ground truth file
        bad_dataset = DataSet(
            name="bad_dataset",
            base_files=test_dataset.base_files,
            query_file=test_dataset.query_file,
            ground_truth_file="nonexistent.ivecs"
        )
        data_source = VecDataSource(bad_dataset)

        with pytest.raises(ValueError, match="Ground truth file not found"):
            data_source.read_neighbors("nonexistent.ivecs")


# Additional tests for edge cases and error handling

def test_read_vectors_corrupted_file(tmp_path):
    """Test reading vectors from a corrupted file raises an error."""
    # Create a corrupted file (not in the correct format)
    corrupted_file = tmp_path / "corrupted.fvecs"
    with open(corrupted_file, 'wb') as f:
        f.write(b'corrupted data')

    # Create a dataset and data source
    dataset = DataSet(
        name="test_dataset",
        base_files=str(tmp_path),
        query_file="query.fvecs",
        ground_truth_file="groundtruth.ivecs"
    )
    data_source = VecDataSource(dataset)

    # Reading from the corrupted file should raise an error
    with pytest.raises(Exception):
        list(data_source.read_vectors(str(corrupted_file)))


def test_dimensions_no_vectors(tmp_path):
    """Test getting dimensions when no vectors are available raises an error."""
    # Create an empty base file with no vectors
    empty_file = tmp_path / "empty.fvecs"
    with open(empty_file, 'wb'):
        pass

    dataset = DataSet(
        name="empty_dataset",
        base_files=[str(empty_file)],
        query_file="query.fvecs",
        ground_truth_file="groundtruth.ivecs"
    )
    data_source = VecDataSource(dataset)

    # Trying to get dimensions should raise an error
    with pytest.raises(ValueError, match="Unable to determine vector dimensions"):
        data_source.get_dimensions()
