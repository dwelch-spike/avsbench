# Tests for AVS Benchmark

This directory contains tests for the AVS Benchmark project.

## Running Tests

To run the tests, you need to have pytest installed:

```bash
pip install pytest
```

Then, from the project root directory, run:

```bash
pytest tests/
```

Or to run a specific test file:

```bash
pytest tests/test_vec_data_source.py
pytest tests/test_client.py
```

To run tests with verbosity and output:

```bash
pytest -v tests/
```

To run only unit tests (excluding integration tests):

```bash
pytest tests/ -k "not integration"
```

To run integration tests that require a running Aerospike server:

```bash
pytest tests/ -m integration
```

## Test Organization

- `test_vec_data_source.py`: Tests for the `VecDataSource` class that handles vector data files
- `test_client.py`: Tests for the client module that handles Aerospike Vector Search client operations
- `test_data/`: Directory containing test data files
- `test_data/create_test_data.py`: Script to generate test data files
- `conftest.py`: Configuration file for pytest that sets up the Python path

## Test Data

The tests use synthetic data files in various vector formats. These are created by the `create_test_data.py` script. 
If you need to regenerate the test data, run:

```bash
python tests/test_data/create_test_data.py
```

## Test Types

### Unit Tests
Most tests are unit tests that mock external dependencies and focus on testing the behavior of individual functions or classes in isolation.

### Integration Tests
Some tests (marked with `@pytest.mark.integration`) require a running Aerospike server. These tests are skipped by default. Run them with `pytest -m integration` when you have a local Aerospike server running. 