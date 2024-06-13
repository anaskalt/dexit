
"""
Manages Peer-to-Peer (P2P) network operations for Federated Learning.

This module contains the P2PHandler class responsible for managing peer-to-peer network interactions
in a federated learning setup. It includes functionality for initializing the network, publishing and
subscribing to messages (including model weights), and maintaining a record of peer statuses.

Classes:
    P2PHandler: Manages the P2P network interactions.

Example:
    # Initialization with network parameters
    p2p_handler = P2PHandler(bootnodes, key_path, topic, packet_size)
    asyncio.run(p2p_handler.init_network())
"""

from typing import List
import asyncio
import libp2p_pyrust as libp2p
from dexit.utils.operations import Operations
from dexit.utils.state import Status, PeerStatus, NetworkState, InferenceRequest, InferenceResult



class P2PHandler:
    """
    Handles initialization and communication over a P2P network for federated learning.

    Attributes:
        bootnodes (list): A list of bootnode addresses for network initialization.
        key_path (str): Path to the keypair used for node identification.
        topic (str): The network topic under which messages are published and subscribed.
        packet_size (int): The size of each packet for breaking down large messages, in bytes.
    """

    def __init__(self, bootnodes, key_path, topic, edge_device_network, server_network, packet_size=1024, device='cpu'):
        self.bootnodes = bootnodes
        self.key_path = key_path
        self.topic = topic
        self.packet_size = packet_size
        self.device = device
        self.local_peer_id = None
        self.local_peer_status = None
        self.network_state = NetworkState()
        self.message_buffer = bytearray()
        self.is_receiving = False
        self.edge_device_network = edge_device_network.to(self.device)
        self.server_network = server_network.to(self.device)

    def init_peer_objects(self, peer_id):
        """Initializes PeerStatus with corresponding peer ID."""
        self.local_peer_status = PeerStatus(peer_id, Status.NONE)
        self.network_state.update_peer_status(self.local_peer_status)

    async def generate_or_load_key(self):
        """
        Generate or load a node keypair for P2P identity.
        Attempts to generate a keypair at the specified path; does nothing if it already exists.
        """
        try:
            libp2p.generate_ed25519_keypair(self.key_path)
        except FileExistsError:
            pass

    async def init_network(self):
        """
        Initialize the P2P network with the specified bootnodes and topic.
        Calls generate_or_load_key to ensure node identity before network initialization.
        """
        await self.generate_or_load_key()
        await libp2p.init_global_p2p_network(self.bootnodes, 0, self.key_path, self.topic)

        self.local_peer_id = await self.get_local_peer_id()
        self.init_peer_objects(self.local_peer_id)

    async def publish_hello(self):
        """
        Publishes a 'hello' message from the local peer to the network.
        This method is useful for announcing the peer's presence to other peers and can be
        extended to include additional metadata about the peer if needed.

        The method handles errors during message publishing and logs success or failure accordingly.
        """
        local_peer_id = await self.get_local_peer_id()
        hello_message = f"hello from {local_peer_id}".encode()

        try:
            await libp2p.publish_message(hello_message)
            print(f"Hello message published from {local_peer_id}.")
        except Exception as e:
            print(f"Failed to publish hello message from {local_peer_id}. Error: {str(e)}")

    async def publish_status(self, peer_status: PeerStatus):
        """
        Publishes the peer's status to the network. This function serializes the PeerStatus object
        and publishes it, allowing other peers in the network to be aware of the local peer's status.

        Args:
            peer_status (PeerStatus): The status object of the local peer.
        """
        operations = Operations(chunk_size=self.packet_size)
        serialized_data = operations.serialize(peer_status)
        await self.publish_objects(serialized_data)

    async def publish_inference_request(self, inference_request: InferenceRequest):
        """
        Publishes the inference request to the network. This function serializes the InferenceRequest object
        and publishes it, enabling other peers in the network to receive and process these requests.

        Args:
            inference_request (InferenceRequest): The request object of the local peer's inference.
        """
        operations = Operations(chunk_size=self.packet_size)
        serialized_data = operations.serialize(inference_request)
        await self.publish_objects(serialized_data)

    async def publish_inference_result(self, inference_result: InferenceResult):
        """
        Publishes the inference result to the network. This function serializes the InferenceResult object
        and publishes it, enabling other peers in the network to receive and utilize these results.

        Args:
            inference_result (InferenceResult): The result object of the local peer's inference.
        """
        operations = Operations(chunk_size=self.packet_size)
        serialized_data = operations.serialize(inference_result)
        await self.publish_objects(serialized_data)

    async def publish_objects(self, serialized_data: List[bytes], delay: float = 0.1):
        """
        Publishes serialized data objects (PeerStatus, InferenceRequest, or InferenceResult) to the P2P network.

        This method serializes the data object (either PeerStatus, InferenceRequest, or InferenceResult) into chunks
        and publishes each chunk to the network, with 'start' and 'end' flags to denote the sequence.
        It introduces an optional delay between sending chunks to prevent network congestion.

        Args:
            serialized_data (List[bytes]): The serialized data chunks to publish.
            delay (float): Delay in seconds between publishing each chunk to prevent network congestion. Defaults to 0.1.
        """
        await libp2p.publish_message(b'start')
        for chunk in serialized_data:
            await libp2p.publish_message(chunk)
            await asyncio.sleep(delay)
        await libp2p.publish_message(b'end')

    '''async def subscribe_to_messages(self):
        def callback_wrapper(message):
            self.message_dispatcher(message)
        
        await libp2p.subscribe_to_messages(callback_wrapper)'''

    async def subscribe_to_messages(self):
        def callback_wrapper(message):
            try:
                self.message_dispatcher(message)
            except Exception as e:
                print(f"Error in message dispatcher: {e}")

        try:
            await libp2p.subscribe_to_messages(callback_wrapper)
        except EOFError as e:
            print(f"EOFError during message subscription: {e}")
        except Exception as e:
            print(f"General error during message subscription: {e}")

    async def start_listening(self):
        """
        Subscribes to messages from the P2P network and dispatches them to the appropriate handlers.
        This method remains effectively unchanged because the modifications to subscribe_to_messages
        ensure that asynchronous message dispatching is handled correctly.
        """
        await self.subscribe_to_messages()

    def message_dispatcher(self, message: bytes):
        """
        Dispatches incoming messages based on their type and content. It handles the start and end of message sequences
        and delegates processing of complete messages.

        Args:
            message (bytes): The received serialized message.
        """
        if message == b'start':
            self.message_buffer.clear()
            self.is_receiving = True
        elif message == b'end' and self.is_receiving:
            self.process_complete_message()
            self.is_receiving = False
        elif self.is_receiving:
            self.message_buffer.extend(message)

    def process_complete_message(self):
        """
        Processes a complete message after it's fully received, from the initial 'start' signal to the 'end' signal. 
        This involves deserializing the message into either a PeerStatus, InferenceRequest, or InferenceResult object and updating the 
        network state accordingly. The method ensures accurate reflection of peer statuses and results in the 
        network's overall state based on incoming data.
        """
        operations = Operations()
        data_object = operations.deserialize([self.message_buffer])
        if isinstance(data_object, PeerStatus):
            self.handle_peer_status_update(data_object)
        elif isinstance(data_object, InferenceRequest):
            self.handle_inference_request_update(data_object)
        elif isinstance(data_object, InferenceResult):
            self.handle_inference_result_update(data_object)
        self.message_buffer.clear()

    def handle_peer_status_update(self, peer_status: PeerStatus):
        """
        Handles the update of a peer's status. After a peer status message is fully received and processed, 
        this function updates the network state with the new status of the peer. It's crucial for maintaining 
        the current state of each peer within the network, facilitating coordinated actions and decisions.
        """
        self.network_state.update_peer_status(peer_status)
        print(f"PeerStatus updated for {peer_status.peer_id}: {peer_status.status}")

    def handle_inference_request_update(self, inference_request: InferenceRequest):
        """
        Handles the update of a peer's inference request. After an inference request message is fully received and processed, 
        this function triggers the processing of the received sample on the local peer.
        """
        print(f"InferenceRequest received from {inference_request.peer_id}")
        asyncio.create_task(self.process_inference_request(inference_request))
        #self.process_inference_request(inference_request)

    def handle_inference_result_update(self, inference_result: InferenceResult):
        """
        Manages the update of a peer's inference result. Upon fully receiving and processing an inference result message, 
        this function updates the network state with the new result of the peer. This action is vital for the 
        distributed inference process, allowing the network to collectively utilize the results based on peer contributions.
        """
        self.network_state.update_inference_result(inference_result)
        print(f"InferenceResult updated for {inference_result.peer_id}.")

    async def process_inference_request(self, inference_request: InferenceRequest):
        try:
            sample = inference_request.sample
            peer_id = inference_request.peer_id

            print(f"Processing inference request from peer {peer_id}")

            # Perform inference using the server network
            output = self.server_network(sample)
            print(f"Server network output shape: {output.shape}")

            # Create an InferenceResult object with the result
            inference_result = InferenceResult(peer_id=peer_id, result=output)

            # Publish the inference result to the network
            print(f"Publishing inference result for peer {peer_id}")
            await self.publish_inference_result(inference_result)

            # Set status to DONE and publish it
            self.local_peer_status.status = Status.DONE
            self.network_state.update_peer_status(self.local_peer_status)
            for _ in range(3):
                await self.publish_status(self.local_peer_status)
            await asyncio.sleep(10)

            # Set status back to READY to ensure continuity
            self.local_peer_status.status = Status.READY
            self.network_state.update_peer_status(self.local_peer_status)
            for _ in range(3):
                await self.publish_status(self.local_peer_status)
            await asyncio.sleep(10)
        except Exception as e:
            print(f"Error processing inference request: {e}")


    async def get_peers(self):
        """
        Retrieves the list of currently connected peers.

        Returns:
            list: A list of peer IDs for all connected peers.
        """
        return await libp2p.get_global_connected_peers()

    async def get_local_peer_id(self):
        """
        Retrieves the local peer ID.

        Returns:
            str: The local peer ID.
        """
        return await libp2p.get_peer_id()

    async def wait_for_peers(self, min_peers=2, check_interval=5):
        """
        Waits for a minimum number of peers to be connected before proceeding.

        Args:
            min_peers (int): The minimum number of peers required to proceed.
            check_interval (int): The interval, in seconds, between checks for connected peers.

        Notes:
            - Continuously checks for the number of connected peers and updates their statuses.
            - Proceeds once the minimum number of peers is connected.
        """
        print(f"Waiting for at least {min_peers} peers to connect...")
        while True:
            current_peers = await self.get_peers()
            if len(current_peers) >= min_peers:
                print(f"Connected peers: {len(current_peers)}. Proceeding.")   
                break
            else:
                print(f"Connected peers: {len(current_peers)}. Waiting...")
                await asyncio.sleep(check_interval)
