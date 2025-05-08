import aerospike_vector_search as avs

from bench.logger import get_logger

logger = get_logger(__name__)


def get_client(host: str, port: int, load_balancer: bool) -> avs.Client:
    return avs.Client(
        seeds=avs.HostPort(host=host, port=port),
        is_loadbalancer=load_balancer
    )


def get_index_name(dataset_name: str, index_id: int) -> str:
    """
    Get the name of an index.

    Args:
        dataset_name: The name of the dataset
        index_id: The ID of the index
    """
    if index_id != -1:
        return f"{dataset_name}-index-{index_id}"
    return f"{dataset_name}-index"


def index_exists(client: avs.Client, index_name: str, namespace: str) -> bool:
    """
    Check if an index exists in the client.

    Args:
        client: The AVS client
        index_name: The name of the index to check
        namespace: The namespace to check the index in
    """
    indexes: list[avs.types.IndexDefinition] = client.index_list()
    return any(index.id.name == index_name and index.id.namespace == namespace for index in indexes)


def _create_index(client: avs.Client, namespace: str, vector_field: str, index_name: str, dimensions: int, distance_metric: avs.VectorDistanceMetric):
    client.index_create(
        namespace=namespace,
        name=index_name,
        vector_field=vector_field,
        dimensions=dimensions,
        vector_distance_metric=distance_metric
    )


def create_index(client: avs.Client, index_name: str, namespace: str, vector_field: str, dimensions: int, distance_metric: avs.VectorDistanceMetric):
    """
    Create a vector index if it doesn't exist.

    Args:
        client: The AVS client
        index_name: The name of the index to create (should be created with get_index_name)
        namespace: The namespace to create the index in
        vector_field: The field name for the vector
        dimensions: The dimensions of the vector
    """
    logger.info(f"Creating index {index_name} in namespace {namespace}")
    if index_exists(client, index_name, namespace):
        logger.info(f"Index {index_name} already exists")
        return
    _create_index(
        client,
        namespace,
        vector_field,
        index_name,
        dimensions,
        distance_metric
    )
    logger.info(f"Index {index_name} created in namespace {namespace}")


def delete_index(client: avs.Client, index_name: str, namespace: str):
    """
    Delete a vector index.

    Args:
        client: The AVS client
        index_name: The name of the index to delete
        namespace: The namespace to delete the index from
    """
    logger.info(f"Deleting index {index_name} from namespace {namespace}")
    client.index_drop(namespace=namespace, name=index_name)
    logger.info(f"Index {index_name} deleted from namespace {namespace}")
