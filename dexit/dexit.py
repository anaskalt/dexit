### TOBE REMOVED BEFORE PACKAGED
import sys

sys.path.insert(0, "../")

import os
import time
import asyncio
import torch
import configparser
import matplotlib.pyplot as plt

from dexit.utils.state import Status, InferenceRequest
from dexit.network.handler import P2PHandler
from dexit.data.dataloaders import testloader

### For debug purposes
ATTEMPTS = 3
torch.autograd.set_detect_anomaly(True)
# Ensure the metrics directory exists
METRICS_DIR = '../metrics'
os.makedirs(METRICS_DIR, exist_ok=True)

# Load configuration
def load_configuration():
    config = configparser.ConfigParser()
    config.read('../conf/node.conf')
    print("Configuration loaded.")
    return config

# Setup P2P network (async)
async def setup_p2p_network(config, edge_device_network, server_network, device):
    p2p_handler = P2PHandler(
        bootnodes=config['P2P']['bootnodes'].split(','),
        key_path=config['P2P']['key_path'],
        topic=config['P2P']['topic'],
        edge_device_network=edge_device_network,
        server_network=server_network,
        packet_size=int(config['P2P']['packet_size']),
        device=device
    )
    return p2p_handler

async def perform_inference(p2p_handler, config, DEVICE):
    """
    Perform the inference process on the edge device.
    """
    # Start inference process
    for data, _ in testloader:
        batch = data.to(DEVICE)
        for sample in batch:
            inference_request = InferenceRequest(peer_id=p2p_handler.local_peer_id, sample=sample)
            
            # Perform local inference
            output, exit_taken = p2p_handler.edge_device_network(sample)
            print(f"Edge device network output shape: {output.shape}, exit taken: {exit_taken}")
            
            if not exit_taken:
                # If exit condition is not met, contact the server
                while True:
                    # Check server availability
                    if p2p_handler.network_state.check_state(Status.READY):
                        print('Server is ready, sending sample for inference')
                        await p2p_handler.publish_inference_request(inference_request)
                        break
                    else:
                        print('Server is not ready, waiting 2 seconds')
                        await asyncio.sleep(2)
                
                # Wait for the server to process the sample and send back the result
                start_time = time.time()
                while not p2p_handler.network_state.check_state(Status.DONE):
                    await asyncio.sleep(2)
                    if time.time() - start_time > int(config['P2P']['infernece_timeout']):
                        print('Timeout waiting for server response')
                        return

                # Process the received inference result
                inference_result = p2p_handler.network_state.get_inference_result(p2p_handler.local_peer_id)
                output = inference_result.result
                print(f"Received inference result from server with shape: {output.shape}")
            
            print(f"Final output shape: {output.shape}")


async def main():
    config = load_configuration()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    edge_device_network = torch.load('models/device.pth').to(DEVICE)
    server_network = torch.load('models/server.pth').to(DEVICE)

    p2p_handler = await setup_p2p_network(config, edge_device_network, server_network, DEVICE)

    await p2p_handler.init_network()

    # Wait for a minimum number of peers to be connected
    await p2p_handler.wait_for_peers(min_peers=1, check_interval=int(config['P2P']['wait_for_peers_interval']))

    # Subscribe to network messages and handle them appropriately
    asyncio.create_task(p2p_handler.subscribe_to_messages())

    # Update and publish the peer's status
    p2p_handler.local_peer_status.status = Status.NONE
    p2p_handler.network_state.update_peer_status(p2p_handler.local_peer_status)

    for _ in range(ATTEMPTS):
        await p2p_handler.publish_status(p2p_handler.local_peer_status)
    await asyncio.sleep(10)

    p2p_handler.local_peer_status.status = Status.JOINED
    p2p_handler.network_state.update_peer_status(p2p_handler.local_peer_status)
    for _ in range(ATTEMPTS):
        await p2p_handler.publish_status(p2p_handler.local_peer_status)

    await asyncio.sleep(10)

    p2p_handler.local_peer_status.status = Status.READY
    p2p_handler.network_state.update_peer_status(p2p_handler.local_peer_status)
    for _ in range(ATTEMPTS):
        await p2p_handler.publish_status(p2p_handler.local_peer_status)

    await asyncio.sleep(10)

    # Determine role and perform actions accordingly
    role = config['ROLE']['role']

    if role == 'edge':
        # Perform inference
        await perform_inference(p2p_handler, config, DEVICE)
    elif role == 'server':
        print("Server waiting for inference requests...")

    # Keep the script running to listen for incoming messages
    while True:
        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
