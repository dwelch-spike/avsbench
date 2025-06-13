import os
import gzip
import numpy as np
from typing import List, Iterator, BinaryIO
import logging
import struct

from bench.data_set import DataSource, DataSet, VectorDataPoint, QueryDataPoint, Neighbors

logger = logging.getLogger(__name__)


class VecDataSource(DataSource):
    """
    A file-based implementation of the DataSource interface.
    Supports .bvec, .fvec, and .ivec file formats with optional gzip compression.
    """

    def __init__(self, dataset: DataSet):
        """
        Initialize a VecDataSource with a dataset configuration.

        Args:
            dataset: The dataset configuration
        """
        logger.debug(f"Initializing VecDataSource with dataset: {dataset}")
        self.dataset = dataset
        self.base_vector_count = None  # Cached base vector count
        self._dimensions = None  # Cached dimensions

    def _is_gzipped(self, file_path: str) -> bool:
        """Check if a file is gzipped by examining its magic number."""
        with open(file_path, 'rb') as f:
            magic = f.read(2)
            if len(magic) != 2:
                return False

            return magic[0] == 0x1f and magic[1] == 0x8b

    def _is_bvecs_format(self, file_path: str) -> bool:
        """Check if a file is in bvecs format based on its name."""
        return 'bvecs' in os.path.basename(file_path).lower()

    def _get_input_stream(self, file_path: str) -> BinaryIO:
        """Get the appropriate input stream for a file, handling gzip if needed."""
        if self._is_gzipped(file_path):
            return gzip.open(file_path, 'rb')
        else:
            return open(file_path, 'rb')

    def _read_int(self, stream: BinaryIO) -> int:
        """Read a 4-byte integer from the stream in little-endian format."""
        data = stream.read(4)
        if len(data) != 4:
            raise ValueError(f"Expected to read 4 bytes, got {len(data)}")
        return struct.unpack('<i', data)[0]

    def _read_float(self, stream: BinaryIO) -> float:
        """Read a 4-byte float from the stream in little-endian format."""
        data = stream.read(4)
        if len(data) != 4:
            raise ValueError(f"Expected to read 4 bytes, got {len(data)}")
        return struct.unpack('<f', data)[0]

    def _read_vectors_from_stream(self, stream: BinaryIO, is_bvecs: bool) -> Iterator[VectorDataPoint]:
        """Read vectors from a stream, yielding VectorDataPoint objects."""
        vector_id = 0

        try:
            while True:
                # Read dimension (first 4 bytes)
                dimension_bytes = stream.read(4)
                if not dimension_bytes or len(dimension_bytes) < 4:
                    break  # End of file

                dimension = struct.unpack('<i', dimension_bytes)[0]

                # Read vector data
                if is_bvecs:
                    # For bvecs, each value is a single byte
                    vector_data = np.frombuffer(stream.read(dimension), dtype=np.uint8).astype(np.float32)
                else:
                    # For fvecs/ivecs, each value is a 4-byte float/int
                    vector_data = np.zeros(dimension, dtype=np.float32)
                    for i in range(dimension):
                        if is_bvecs:
                            vector_data[i] = float(stream.read(1)[0])
                        else:
                            vector_data[i] = self._read_float(stream)

                yield VectorDataPoint(id=vector_id, vector=vector_data)
                vector_id += 1

        except Exception as e:
            logger.error(f"Error reading vector data: {e}")
            raise

    def read_vectors(self, file_path: str) -> Iterator[VectorDataPoint]:
        """Read vectors from the specified file path."""
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        is_bvecs = self._is_bvecs_format(file_path)

        with self._get_input_stream(file_path) as stream:
            yield from self._read_vectors_from_stream(stream, is_bvecs)

    def read_base_vectors(self, file_index: int) -> Iterator[VectorDataPoint]:
        """Read and return the base vectors used for indexing."""

        try:
            base_file = self.dataset.base_files[file_index]
        except IndexError:
            raise ValueError(f"Base file index {file_index} out of range, expected range is 0 to {len(self.dataset.base_files) - 1}")

        if not os.path.exists(base_file):
            logger.error(f"Base file {base_file} not found")
            raise ValueError(f"Base file {base_file} not found")

        logger.info(f"Reading base file {base_file}")
        yield from self.read_vectors(base_file)

    def read_query_vectors(self) -> Iterator[QueryDataPoint]:
        """Read and return the query vectors with ground truth neighbors."""
        query_path = self.dataset.query_file
        if not os.path.exists(query_path):
            raise ValueError(f"Query file not found: {query_path}")

        query_vectors = self.read_vectors(query_path)
        neighbors = self.read_neighbors(self.dataset.ground_truth_file)

        # Match query vectors with their neighbors
        for i, vector in enumerate(query_vectors):
            if i < len(neighbors):
                yield QueryDataPoint(
                    vector=vector.vector,
                    neighbors=neighbors[i].neighbors
                )
            else:
                # If we don't have neighbors for this query, use empty list
                logger.warning(f"No neighbors found for query {i}")
                yield QueryDataPoint(
                    vector=vector.vector,
                    neighbors=[]
                )

    def get_dimensions(self) -> int:
        """Return the dimensionality of the vectors."""
        if self._dimensions is not None:
            return self._dimensions

        # Try to read the first vector to determine dimensions
        base_file = self.dataset.base_files[0]

        if base_file and os.path.exists(base_file):
            try:
                vector = next(self.read_vectors(base_file))
                if vector:
                    self._dimensions = len(vector.vector)
                    return self._dimensions
            except StopIteration:
                pass

        raise ValueError("Unable to determine vector dimensions - no vectors found")

    def get_num_base_vectors(self) -> int:
        """Return the total number of base vectors."""
        if self.base_vector_count is None:
            self.base_vector_count = len(list(self.read_base_vectors(0)))
        return self.base_vector_count

    def get_vector_distance_metric(self) -> str:
        """Return the appropriate distance metric for this dataset."""
        # Match the Kotlin implementation's default
        return "squared_euclidean"

    def read_neighbors(self, file_path: str) -> List[Neighbors]:
        """
        Read ground truth neighbors data for evaluating query results.

        Returns:
            List of Neighbors objects containing query ID and its neighbor IDs
        """
        if not os.path.exists(file_path):
            raise ValueError(f"Ground truth file not found: {file_path}")

        neighbors_list = []

        with self._get_input_stream(file_path) as stream:
            vector_id = 0

            try:
                while True:
                    # Read dimension (number of neighbors)
                    dimension_bytes = stream.read(4)
                    if not dimension_bytes or len(dimension_bytes) < 4:
                        break  # End of file

                    dimension = struct.unpack('<i', dimension_bytes)[0]

                    # Read neighbor IDs
                    neighbor_ids = []
                    for _ in range(dimension):
                        neighbor_id = self._read_int(stream)
                        neighbor_ids.append(neighbor_id)

                    neighbors_list.append(Neighbors(id=vector_id, neighbors=neighbor_ids))
                    vector_id += 1

            except Exception as e:
                logger.error(f"Error reading neighbors data: {e}")
                raise

        return neighbors_list

    # def copy_vectors(self, input_file: str, output_file: str, limit: int = None):
    #     """
    #     Copy vectors from input file to output file, optionally limiting the count.
    #     Similar to the Kotlin implementation's copyVectors function.
    #     """
    #     if not os.path.exists(input_file):
    #         raise ValueError(f"Input file not found: {input_file}")

    #     is_bvecs = self._is_bvecs_format(input_file)

    #     with self._get_input_stream(input_file) as input_stream, open(output_file, 'wb') as output_stream:
    #         vector_id = 0

    #         while True:
    #             # Check if we've reached the limit
    #             if limit is not None and vector_id >= limit:
    #                 break

    #             # Report progress
    #             if vector_id > 0 and vector_id % 100_000 == 0:
    #                 logger.info(f"Copied {vector_id} vectors")

    #             # Read dimension
    #             input_stream.seek(input_stream.tell())  # Mark current position
    #             dimension_bytes = input_stream.read(4)
    #             if not dimension_bytes or len(dimension_bytes) < 4:
    #                 break  # End of file

    #             dimension = struct.unpack('<i', dimension_bytes)[0]

    #             # Calculate vector size in bytes
    #             vector_bytes = 4  # dimension int
    #             if is_bvecs:
    #                 vector_bytes += dimension  # byte array
    #             else:
    #                 vector_bytes += 4 * dimension  # float array

    #             # Reset to start of vector
    #             input_stream.seek(input_stream.tell() - 4)

    #             # Copy the entire vector
    #             buffer = input_stream.read(vector_bytes)
    #             output_stream.write(buffer)

    #             vector_id += 1

    #         logger.info(f"Copied {vector_id} vectors")
