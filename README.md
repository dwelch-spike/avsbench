# AVS Python Client Benchmark

This project provides a benchmarking tool for AVS (Alexa Voice Service) using the AVS Python client. It is designed to be similar in functionality to the AVS Kotlin benchmark.

## Setup

1.  **Create and activate a Python virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare the dataset:**

    A sample dataset (`siftsmall`) is provided in the `datasets/` directory. You'll need to extract it:

    ```bash
    cd datasets
    tar -xzf siftsmall.tar.gz
    cd ..
    ```
    This will create a `datasets/siftsmall` directory.

## Running the Benchmark

The main script for running the benchmark is `main.py`. It requires the path to a dataset and offers various other options to configure the AVS cluster, index, and benchmark behavior.

**Basic usage:**

```bash
python main.py --base-folder datasets/siftsmall
```

**Available options:**

You can see all available options by running:

```bash
python main.py --help
```

Key options include:

*   **AVS Cluster Options:**
    *   `--host <hostname>`: Host of the AVS cluster (default: `localhost`).
    *   `--port <port_number>`: Port of the AVS cluster (default: `5000`).
    *   `--load-balancer`: Flag to indicate if the host is a load balancer.
*   **Index Options:**
    *   `--namespace <namespace_name>`: Namespace for the index (default: `test`).
    *   `--vector-index-id <id>`: ID of the index.
    *   `--vector-distance-metric <metric>`: Distance metric (e.g., `SQUARED_EUCLIDEAN`).
*   **Dataset Options:**
    *   `--base-folder <path>`: (Required) Path to the folder containing dataset files.
    *   `--base-file-count <count>`: Number of base dataset files to load.
    *   `--base-vector-count <count>`: Number of vectors to load.
*   **Functionality Options:**
    *   `--no-insert-records`: Skip inserting records.
    *   `--no-query-record`: Skip querying records.

## Running Tests

Tests are managed using `pytest`. The configuration is in `pytest.ini`, and test files are located in the `tests/` directory.

Some tests are marked as `integration` tests and require a running Aerospike server.

*   **Run all tests (including integration tests):**

    ```bash
    pytest
    ```

*   **Run only unit tests (excluding integration tests):**

    ```bash
    pytest -k "not integration"
    ```

*   **Run only integration tests:**

    ```bash
    pytest -m integration
    ```
