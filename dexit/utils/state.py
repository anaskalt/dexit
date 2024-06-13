from enum import Enum, auto

class Status(Enum):
    """
    Defines possible statuses for peers within a Decentralized Federated Learning (DFL) network.
    
    Attributes:
        JOINED: Indicates a peer has successfully joined the network.
        READY: Suggests that a peer has completed its current training cycle and is ready.
        BUSY: Indicates that a peer is currently processing data.
        WAITING: Indicates that a peer is waiting for resources or another node.
        NOT_READY: Indicates that a peer is not ready to process data.
        DONE: Indicates that a peer has completed its tasks.
        ERROR: Denotes that an error has occurred in a peer's training process or network communication.
        NONE: An initial or reset state, indicating that the peer's current status is not set or has been cleared.
    """
    JOINED = auto()
    READY = auto()
    BUSY = auto()
    WAITING = auto()
    NOT_READY = auto()
    DONE = auto()
    ERROR = auto()
    NONE = auto()

class PeerStatus:
    """Represents a single peer's status."""
    def __init__(self, peer_id: str, status: Status):
        self._peer_id = peer_id
        self._status = status

    @property
    def peer_id(self) -> str:
        """Gets the peer's ID."""
        return self._peer_id

    @peer_id.setter
    def peer_id(self, value: str):
        """Sets the peer's ID."""
        self._peer_id = value

    @property
    def status(self) -> Status:
        """Gets the peer's status."""
        return self._status

    @status.setter
    def status(self, value: Status):
        """Sets the peer's status."""
        self._status = value

class InferenceRequest:
    """Represents an inference request with a sample."""
    def __init__(self, peer_id: str, sample):
        self._peer_id = peer_id
        self._sample = sample

    @property
    def peer_id(self) -> str:
        """Gets the peer's ID."""
        return self._peer_id

    @peer_id.setter
    def peer_id(self, value: str):
        """Sets the peer's ID."""
        self._peer_id = value

    @property
    def sample(self):
        """Gets the sample."""
        return self._sample

    @sample.setter
    def sample(self, value):
        """Sets the sample."""
        self._sample = value

class InferenceResult:
    """Represents the result of an inference."""
    def __init__(self, peer_id: str, result):
        self._peer_id = peer_id
        self._result = result

    @property
    def peer_id(self) -> str:
        """Gets the peer's ID."""
        return self._peer_id

    @peer_id.setter
    def peer_id(self, value: str):
        """Sets the peer's ID."""
        self._peer_id = value

    @property
    def result(self):
        """Gets the result."""
        return self._result

    @result.setter
    def result(self, value):
        """Sets the result."""
        self._result = value


class NetworkState:
    """Encapsulates the network state, managing statuses and inference results of all peers."""
    def __init__(self):
        self.peer_statuses = {}
        self.inference_results = {}

    def update_peer_status(self, peer_status_info: PeerStatus):
        """Updates or adds the status of a peer."""
        self.peer_statuses[peer_status_info.peer_id] = peer_status_info.status

    def remove_peer(self, peer_id: str):
        """Removes a peer from the network state."""
        self.peer_statuses.pop(peer_id, None)
        self.inference_results.pop(peer_id, None)

    def reset_network_state(self):
        """Resets the network state to its initial condition."""
        self.peer_statuses.clear()
        self.inference_results.clear()

    def update_inference_result(self, result: InferenceResult):
        """Updates or adds an inference result of a peer."""
        self.inference_results[result.peer_id] = result.result

    def check_state(self, *states) -> bool:
        """Checks if all peers are in any of the specified states.
        
        Args:
            *states (Status): Variable number of state arguments to check against.
        
        Returns:
            bool: True if all peers are in any of the specified states, False otherwise.
        """
        return all(status in states for status in self.peer_statuses.values())

    def get_network_summary(self) -> dict:
        """Returns a summary of the network state."""
        statuses_summary = {peer_id: status.name for peer_id, status in self.peer_statuses.items()}
        return {
            "statuses": statuses_summary,
            "inference_results": self.inference_results
        }

    def get_all_results(self) -> dict:
        """Returns the inference results of all peers."""
        return self.inference_results.copy()
