import pickle
from typing import List, Union
from dexit.utils.state import PeerStatus, InferenceRequest, InferenceResult
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Operations:
    """
    A class for performing operations on peer statuses, inference requests, and inference results,
    including serialization and deserialization. Supports chunking for efficient network transmission.

    Attributes:
        chunk_size (int): Size of chunks for serialized data, enabling efficient network transmission.
                          None means no chunking.
    """

    def __init__(self, chunk_size: int = None):
        """
        Initializes the Operations class with optional chunking support.

        Args:
            chunk_size (int, optional): The size of each chunk for serialized data.
                                        Defaults to None, indicating no chunking.
        """
        self.chunk_size = chunk_size

    def serialize(self, data_object: Union[PeerStatus, InferenceRequest, InferenceResult]) -> List[bytes]:
        """
        Serializes a data object (PeerStatus, InferenceRequest, or InferenceResult) into bytes,
        optionally splitting into chunks.

        Args:
            data_object (Union[PeerStatus, InferenceRequest, InferenceResult]): The object to serialize.

        Returns:
            List[bytes]: The serialized data as a list of bytes, chunked if specified.
        """
        serialized_data = pickle.dumps(data_object)
        logging.debug(f"Serialized data size: {len(serialized_data)} bytes")
        if self.chunk_size is not None:
            chunks = [serialized_data[i:i + self.chunk_size] for i in range(0, len(serialized_data), self.chunk_size)]
            logging.debug(f"Data chunked into {len(chunks)} parts")
            return chunks
        else:
            return [serialized_data]

    def deserialize(self, serialized_data: List[bytes]) -> Union[PeerStatus, InferenceRequest, InferenceResult]:
        """
        Deserializes a list of bytes back into a data object (PeerStatus, InferenceRequest, or InferenceResult).

        Args:
            serialized_data (List[bytes]): The serialized data chunks.

        Returns:
            Union[PeerStatus, InferenceRequest, InferenceResult]: The deserialized data object.
        """
        data_bytes = b''.join(serialized_data)
        logging.debug(f"Deserializing data of size: {len(data_bytes)} bytes")
        
        deserialized_data = pickle.loads(data_bytes)
        logging.debug(f"Deserialized data type: {type(deserialized_data)}")
        return deserialized_data
