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
    def __init__(self, peer_id: str, status: Status, role: str):
        self._peer_id = peer_id
        self._status = status
        self._role = role

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

    @property
    def role(self) -> str:
        """Gets the peer's role."""
        return self._role

    @role.setter
    def role(self, value: str):
        """Sets the peer's role."""
        self._role = value

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
    def __init__(self, peer_id: str, result, exit_point: str):
        self._peer_id = peer_id
        self._result = result
        self._exit_point = exit_point

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

    @property
    def exit_point(self) -> str:
        return self._exit_point

    @exit_point.setter
    def exit_point(self, value):
        """Sets the exit_point."""
        self._exit_point = value

class NetworkState:
    """Encapsulates the network state, managing statuses and inference results of all peers."""
    def __init__(self):
        self.peer_statuses = {}
        self.inference_results = {}
        self.inference_requests = []

    def update_peer_status(self, peer_status_info: PeerStatus):
        """Updates or adds the status and the role of a peer."""
        self.peer_statuses[peer_status_info.peer_id] = {
            'status': peer_status_info.status,
            'role': peer_status_info.role
        }

    def get_peer_by_role(self, role: str) -> str:
        for peer_id, info in self.peer_statuses.items():
            if info['role'] == role:
                return peer_id
        return None

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

    def get_inference_result(self, peer_id: str):
        return self.inference_results.get(peer_id)

    def update_inference_request(self, request: InferenceRequest):
        self.inference_requests.append(request)

    def get_inference_request(self):
        return self.inference_requests.pop(0) if self.inference_requests else None

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
        return {
            "statuses": {peer_id: info['status'].name for peer_id, info in self.peer_statuses.items()},
            "roles": {peer_id: info['role'] for peer_id, info in self.peer_statuses.items()},
            "inference_results": self.inference_results
        }

    '''def get_network_summary(self) -> dict:
        return {
            "statuses": {peer_id: info['status'].name for peer_id, info in self.peer_statuses.items()},
            "roles": {peer_id: info['role'] for peer_id, info in self.peer_statuses.items()},
            "inference_results": self.inference_results
        }'''
    
    def get_all_results(self) -> dict:
        """Returns the inference results of all peers."""
        return self.inference_results.copy()
