from dataclasses import dataclass
from typing import List, Iterator
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class VectorDataPoint:
    """A vector data point with ID and vector data."""
    id: int
    vector: np.ndarray


@dataclass
class QueryDataPoint:
    """A query vector with its corresponding neighbors."""
    vector: np.ndarray
    neighbors: List[int]


@dataclass
class Neighbors:
    """Ground truth neighbors for a query vector."""
    id: int
    neighbors: List[int]


@dataclass
class DataSet:
    name: str
    base_files: List[str]
    query_file: str
    ground_truth_file: str


class DataSource(ABC):
    @abstractmethod
    def read_base_vectors(self, file_index: int) -> Iterator[VectorDataPoint]:
        """Read and return the base vectors used for indexing."""
        pass

    @abstractmethod
    def read_query_vectors(self) -> Iterator[QueryDataPoint]:
        """Read and return the query vectors used for evaluation."""
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """Return the dimensionality of the vectors."""
        pass

    @abstractmethod
    def get_num_base_vectors(self) -> int:
        """Return the total number of base vectors."""
        pass

    @abstractmethod
    def read_vectors(self, file_path: str) -> Iterator[VectorDataPoint]:
        """Read vectors from the specified file path."""
        pass

    @abstractmethod
    def read_neighbors(self, file_path: str) -> List[Neighbors]:
        """Read neighbors from the specified file path."""
        pass

    @abstractmethod
    def get_vector_distance_metric(self) -> str:
        """Return the appropriate distance metric for this dataset."""
        pass
