#!/usr/bin/env python3
"""
Create test data for vector dataset testing.
This script creates test vector files in fvecs, ivecs, and bvecs formats.
"""
import os
import numpy as np
import struct
import gzip


def write_fvecs(vectors, filename):
    """Write vectors to a file in fvecs format."""
    with open(filename, 'wb') as f:
        for vector in vectors:
            dimension = len(vector)
            f.write(struct.pack('<i', dimension))
            f.write(struct.pack(f'<{dimension}f', *vector))


def write_ivecs(vectors, filename):
    """Write vectors to a file in ivecs format."""
    with open(filename, 'wb') as f:
        for vector in vectors:
            dimension = len(vector)
            f.write(struct.pack('<i', dimension))
            # Convert to int for ivecs format
            f.write(struct.pack(f'<{dimension}i', *[int(x) for x in vector]))


def write_bvecs(vectors, filename):
    """Write vectors to a file in bvecs format."""
    with open(filename, 'wb') as f:
        for vector in vectors:
            dimension = len(vector)
            f.write(struct.pack('<i', dimension))
            # Convert to byte for bvecs format
            f.write(struct.pack(f'<{dimension}B', *[min(255, max(0, int(x))) for x in vector]))


def write_neighbors(neighbors, filename):
    """Write neighbors to a file in the ground truth format."""
    with open(filename, 'wb') as f:
        for neighbor_list in neighbors:
            dimension = len(neighbor_list)
            f.write(struct.pack('<i', dimension))
            f.write(struct.pack(f'<{dimension}i', *neighbor_list))


def gzip_file(input_filename, output_filename):
    """Gzip a file."""
    with open(input_filename, 'rb') as f_in:
        with gzip.open(output_filename, 'wb') as f_out:
            f_out.write(f_in.read())


def create_test_data():
    """Create test data for vector dataset testing."""
    os.makedirs('tests/test_data', exist_ok=True)

    # Create base vectors (3 files with 10 vectors each, 4 dimensions)
    for i in range(3):
        base_vectors = np.random.rand(10, 4) * 100

        # Write in different formats
        write_fvecs(base_vectors, f'tests/test_data/base_{i}.fvecs')
        write_ivecs(base_vectors, f'tests/test_data/base_{i}.ivecs')
        write_bvecs(base_vectors, f'tests/test_data/base_{i}.bvecs')

        # Also create a gzipped version
        gzip_file(f'tests/test_data/base_{i}.fvecs', f'tests/test_data/base_{i}.fvecs.gz')

    # Create query vectors (5 vectors, 4 dimensions)
    query_vectors = np.random.rand(5, 4) * 100
    write_fvecs(query_vectors, 'tests/test_data/query.fvecs')

    # Create ground truth (neighbors for each query vector)
    neighbors = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7]]
    write_neighbors(neighbors, 'tests/test_data/groundtruth.ivecs')


if __name__ == "__main__":
    create_test_data()
    print("Test data created successfully in tests/test_data/")
