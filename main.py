import argparse
import os
import re
import time
import aerospike_vector_search as avs

from bench.logger import get_logger
from bench.client import get_client, create_index, get_index_name
from bench.data_set import DataSet, DataSource
from bench.vec_data_source import VecDataSource

logger = get_logger(__name__)


VECTOR_FIELD = "vector"


# Handle argument parsing
# PLANS
# We will initially support...
# AVS Cluster args
# - --host -H
# - --port -p
# - --load-balancer -L
# NOTE: a big difference between this and the kotlin benchmarks:
# The kotlin benchmarks don't use a load-balancer option. They do everything
# with cluster tending and accept a listeners argument for proper configuration.


def add_cluster_args(parser):
    group = parser.add_argument_group('AVS Cluster Options')
    group.add_argument("--host", "-H", type=str, required=False, default="localhost",
                       help="The host of the AVS cluster")
    group.add_argument("--port", "-p", type=int, required=False, default=5000,
                       help="The port of the AVS cluster")
    group.add_argument("--load-balancer", "-L", type=bool, required=False, default=False,
                       help="Is the host address a load balancer?")


# Index args
# - --namespace -n
# - --vector-index-id -V
# - --vector-distance-metric -d

def add_index_args(parser):
    group = parser.add_argument_group('Index Options')
    group.add_argument("--namespace", "-n", type=str, required=False, default="test",
                       help="The namespace to write vector data to and to index")
    group.add_argument("--vector-index-id", "-V", type=int, required=False, default=-1,
                       help="The id of the index")
    group.add_argument("--vector-distance-metric", "-d", type=lambda x: avs.VectorDistanceMetric[x.upper()], required=False, default=avs.VectorDistanceMetric.SQUARED_EUCLIDEAN,
                       help="The distance metric to use for vector search, options are: " + ", ".join(avs.VectorDistanceMetric.__members__.keys()))


# Dataset args
# - --base-folder -f
# - --base-file-count -fc
# - --base-vector-count -v

def add_dataset_args(parser):
    group = parser.add_argument_group('Dataset Options')
    group.add_argument("--base-folder", "-f", type=str, required=True,
                       help="The base folder containing dataset files")
    group.add_argument("--base-file-count", "-fc", type=int, required=False, default=1,
                       help="How many dataset files to load")
    group.add_argument("--base-vector-count", "-v", type=int, required=False, default=0,
                       help="How many vectors to load")
    # group.add_argument("--query-file", "-qf", type=str, required=False, default="query.npy",
    #                    help="Name of the query vector file")
    # group.add_argument("--ground-truth-file", "-gt", type=str, required=False, default="groundtruth.npy",
    #                    help="Name of the ground truth file")


# Functionality args
# - --no-insert-records -x
# - --no-query-record -q

def add_functionality_args(parser):
    group = parser.add_argument_group('Functionality Options')
    group.add_argument("--no-insert-records", "-x", action="store_true",
                       help="Don't write any records")
    group.add_argument("--no-query-record", "-q", action="store_true",
                       help="Don't perform any queries")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AVS Benchmark")

    # Add all argument groups
    add_cluster_args(parser)
    add_index_args(parser)
    add_dataset_args(parser)
    add_functionality_args(parser)

    return parser.parse_args()


def get_dataset(dataset_name: str, base_folder: str) -> DataSet:
    base_files = get_base_files(base_folder, dataset_name)
    if not base_files:
        raise ValueError(f"No base files found for dataset {dataset_name} in {base_folder}")
    logger.info(f"Found {len(base_files)} base files for dataset {dataset_name} in {base_folder}")

    query_file = get_query_file(base_folder, dataset_name)
    if not query_file:
        raise ValueError(f"No query file found for dataset {dataset_name} in {base_folder}")
    logger.info(f"Found query file {query_file} for dataset {dataset_name} in {base_folder}")

    ground_truth_file = get_ground_truth_file(base_folder, dataset_name)
    if not ground_truth_file:
        raise ValueError(f"No ground truth file found for dataset {dataset_name} in {base_folder}")
    logger.info(f"Found ground truth file {ground_truth_file} for dataset {dataset_name} in {base_folder}")

    return DataSet(
        name=dataset_name,
        base_files=base_files,
        query_file=query_file,
        ground_truth_file=ground_truth_file
    )


def get_datasource(
    dataset: DataSet
) -> DataSource:
    """
    Load the dataset for benchmarking.

    Args:
        dataset: The dataset to load

    Returns:
        A VecDataSource instance initialized with the dataset
    """
    logger.info(f"Loading dataset from {dataset}")

    data_source = VecDataSource(dataset)
    dimensions = data_source.get_dimensions()
    num_vectors = data_source.get_num_base_vectors()
    logger.info(f"Loaded dataset with {num_vectors} vectors of {dimensions} dimensions")
    return data_source


def insert_vectors(client: avs.Client, data_source: DataSource, data_set: DataSet, namespace: str, dataset_name: str, vector_field: str):
    """
    Insert vectors from the data source into the index.

    Args:
        client: The AVS client
        data_source: The data source containing the vectors to insert
        data_set: The dataset containing the vectors to insert
        namespace: The namespace containing the index
        dataset_name: The name of the dataset
        vector_field: The field name of the vector in the dataset
    """
    logger.info("Writing vectors")    
    total_inserted = 0
    total_time = 0.0
    # TODO: parallelize this
    for file_index in range(len(data_set.base_files)):
        vector_generator = data_source.read_base_vectors(file_index)
        inserted = 0
        start_time = time.time()
        for vector in vector_generator:
            key = f"{dataset_name}-{vector.id}"
            try:
                client.upsert(namespace=namespace, key=key, record_data={"id": vector.id, vector_field: vector.vector})
                inserted += 1
            except Exception as e:
                logger.error(f"Error inserting vector {vector.id}: {e}")
        end_time = time.time()
        total_time += end_time - start_time
        total_inserted += inserted
    logger.info(f"Inserted {total_inserted} total records in {total_time} seconds")


def query_vectors(client: avs.Client, data_source: DataSource, data_set: DataSet, namespace: str, index_name: str):
    """
    Query vectors from the index for benchmarking.

    Args:
        client: The AVS client
        data_source: The data source containing the query vectors
        namespace: The namespace containing the index
        index_name: The name of the index

    Returns:
        Benchmark results
    """
    logger.info("Querying vectors")
    query_generator = data_source.read_vectors(data_set.query_file)
    neighbors = data_source.read_neighbors(data_set.ground_truth_file)
    intersection_total = 0
    expected_total = 0
    successful_queries = 0
    error_queries = 0
    total_time = 0.0
    for query in query_generator:
        # TODO: allow specifying topk results limit
        start_time = time.time()
        try:
            actual_neighbors = client.vector_search(namespace=namespace, index_name=index_name, query=query.vector)
            successful_queries += 1
        except Exception as e:
            error_queries += 1
            logger.error(f"Error querying vector {query.id}: {e}")
        end_time = time.time()
        total_time += end_time - start_time

        # TODO: get topk neighbors instead of hardcoding to 10
        expected_neighbors = neighbors[query.id].neighbors[:10]
        for actual_neighbor in actual_neighbors:
            if actual_neighbor.fields["id"] in expected_neighbors:
                intersection_total += 1
        expected_total += len(expected_neighbors)

    recall = (intersection_total / expected_total) * 100

    logger.info(f"Queried {successful_queries} successful queries and {error_queries} errors in {total_time} seconds, recall: {recall}%")


# TODO: typically there are only 1 query and ground truth file so we don't need the number on the end
# maybe this case should be handled in a separate function
def validate_dataset_file(file_name: str, file_type: str, dataset_name: str) -> bool:
    file_pattern = re.compile(
        f"^{re.escape(dataset_name)}_{file_type}(_\\d+)?\\..*$"
    )
    return file_pattern.match(file_name) is not None


def validate_query_file(file_name: str, dataset_name: str) -> bool:
    # pattern to match files like siftsmall_query.fvecs
    return validate_dataset_file(file_name, "query", dataset_name)


def get_query_file(base_folder: str, dataset_name: str) -> str:
    for file_name in os.listdir(base_folder):
        if validate_query_file(file_name, dataset_name):
            return os.path.join(base_folder, file_name)
    return None


def validate_ground_truth_file(file_name: str, dataset_name: str) -> bool:
    # pattern to match files like siftsmall_groundtruth.ivecs
    return validate_dataset_file(file_name, "groundtruth", dataset_name)


def get_ground_truth_file(base_folder: str, dataset_name: str) -> str:
    for file_name in os.listdir(base_folder):
        if validate_ground_truth_file(file_name, dataset_name):
            return os.path.join(base_folder, file_name)
    return None


def validate_base_file(file_name: str, dataset_name: str) -> bool:
    # pattern to match files like siftsmall_base_0.fvecs
    return validate_dataset_file(file_name, "base", dataset_name)


def get_base_files(base_folder: str, dataset_name: str) -> list[str]:
    base_files = []
    for file_name in os.listdir(base_folder):
        if validate_base_file(file_name, dataset_name):
            base_files.append(os.path.join(base_folder, file_name))
    return base_files


def main():
    args = parse_args()

    # Run benchmark
    # TODO: Implement the benchmark logic here
    # the flow should be:
    # 1. create an aerospike-vector-search client and connect to the cluster
    # 2. ensure the dataset is present and start loading it
    # 3. create the index if it doesn't exist
    # 4. insert the vectors into the index
    # 5. query the vectors in the index
    # 6. print the results

    client = get_client(args.host, args.port, args.load_balancer)

    base_folder = os.path.normpath(args.base_folder)
    dataset_name = os.path.basename(base_folder)
    logger.info(f"Using dataset {dataset_name} from {base_folder}")

    dataset = get_dataset(dataset_name, base_folder)
    logger.debug(f"Got dataset: {dataset}")

    datasource = get_datasource(dataset)
    logger.debug(f"Got datasource: {datasource}")

    index_name = get_index_name(args.namespace, args.vector_index_id)
    logger.debug(f"Got index name: {index_name}")

    vector_field = VECTOR_FIELD
    logger.debug(f"Using vector field: {vector_field}")

    distance_metric = args.vector_distance_metric
    create_index(client, index_name, args.namespace, vector_field, datasource.get_dimensions(), distance_metric)
    if not args.no_insert_records:
        insert_vectors(client, datasource, dataset, args.namespace, dataset_name, vector_field)
    else:
        logger.info("Skipping vector insertion")
    if not args.no_query_record:
        query_vectors(client, datasource, dataset, args.namespace, index_name)
    else:
        logger.info("Skipping queries")

    # Stats Notes
    # We will collect the following stats:
    # INDEX CREATION PHASE
    # INSERT PHASE
    # - Total Time
    # - Finished Insert Count
    # QUERY PHASE
    # - Successful Query Count
    # - Error Query Count
    # - Total Time
    # - Queries Recall
    logger.info("Benchmark completed successfully")


if __name__ == "__main__":
    main()
