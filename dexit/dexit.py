### TOBE REMOVED BEFORE PACKAGED
import sys
import os

#sys.path.insert(0, "../")

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'early_exit'))

import logging
import asyncio
import torch
import configparser
import matplotlib.pyplot as plt

from dexit.utils.state import Status, PeerStatus, InferenceRequest, InferenceResult
from dexit.network.handler import P2PHandler
from dexit.data.dataloaders import CIFARDataLoader
#from dexit.early_exit.early_exit import * 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

### For debug purposes
ATTEMPTS = 3
torch.autograd.set_detect_anomaly(True)
# Ensure the metrics directory exists
METRICS_DIR = '../metrics'
os.makedirs(METRICS_DIR, exist_ok=True)

def load_configuration():
    config = configparser.ConfigParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'conf', 'node.conf')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config.read(config_path)
    return config

def load_model(path, device):
    try:
        model = torch.load(path, map_location=device)
        logging.info(f"Successfully loaded model from {path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {path}: {str(e)}")
        return None

'''def load_configuration():
    config = configparser.ConfigParser()
    config.read('conf/node.conf')
    return config'''

async def setup_p2p_network(config, models, device):
    p2p_handler = P2PHandler(
        bootnodes=config['P2P']['bootnodes'].split(','),
        key_path=config['P2P']['key_path'],
        topic=config['P2P']['topic'],
        models=models,
        packet_size=int(config['P2P']['packet_size']),
        device=device,
        role=config['ROLE']['role'] 
    )
    return p2p_handler

def print_network_state(p2p_handler):
    logging.info("Current Network State:")
    for peer_id, info in p2p_handler.network_state.peer_statuses.items():
        logging.info(f"Peer ID: {peer_id}, Role: {info['role']}, Status: {info['status']}")

async def perform_edge_inference(p2p_handler, timeout, device):
    dataloader = CIFARDataLoader().get_loader()
    for data, _ in dataloader:
        batch = data.to(device)
        for sample in batch:
            output, exit_taken = p2p_handler.models['edge_device'](sample)
            
            if not exit_taken:
                cloud1_peer_id = p2p_handler.network_state.get_peer_by_role('cloud1')
                if cloud1_peer_id:
                    inference_request = InferenceRequest(peer_id=p2p_handler.local_peer_id, sample=output)
                    await p2p_handler.send_inference_request(cloud1_peer_id, inference_request)
                    
                    result = await p2p_handler.wait_for_inference_result(peer_id=p2p_handler.local_peer_id, timeout=timeout)
                    if result:
                        output = result.result
                    else:
                        logging.warning("Cloud1 inference timeout")
                else:
                    logging.warning("Cloud1 peer not found")
            
            logging.info(f"Final output shape: {output.shape}")

async def perform_cloud1_inference(p2p_handler, config, device):
    while True:
        inference_request = await p2p_handler.wait_for_inference_request(timeout=60)
        if inference_request:
            sample = inference_request.sample.to(device)
            output, exit_taken = p2p_handler.models['cloud1'](sample)
            
            result = InferenceResult(peer_id=inference_request.peer_id, result=output)
            await p2p_handler.send_inference_result(inference_request.peer_id, result)
            
            if not exit_taken:
                cloud2_peer_id = p2p_handler.network_state.get_peer_by_role('cloud2')
                if cloud2_peer_id:
                    new_request = InferenceRequest(peer_id=inference_request.peer_id, sample=output)
                    await p2p_handler.send_inference_request(cloud2_peer_id, new_request)
                else:
                    logging.warning("Cloud2 peer not found")
        else:
            await asyncio.sleep(0.1)

async def perform_cloud2_inference(p2p_handler, config, device):
    while True:
        inference_request = await p2p_handler.wait_for_inference_request(timeout=60)
        if inference_request:
            sample = inference_request.sample.to(device)
            output, _ = p2p_handler.models['cloud2'](sample)
            
            result = InferenceResult(peer_id=inference_request.peer_id, result=output)
            await p2p_handler.send_inference_result(inference_request.peer_id, result)
        else:
            await asyncio.sleep(0.1)

async def main():
    try:
        config = load_configuration()
        logging.info("Configuration loaded successfully")
        
        for section in config.sections():
            logging.info(f"Section: {section}")
            for key, value in config[section].items():
                logging.info(f"  {key} = {value}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        models = {}
        for key in ['edge_device', 'cloud1', 'cloud2']:
            model_path = config['MODELS'][key]
            model = load_model(model_path, device)
            if model is not None:
                models[key] = model
            else:
                logging.error(f"Failed to load model for {key}")
                return

        p2p_handler = await setup_p2p_network(config, models, device)

        await p2p_handler.init_network()

        await p2p_handler.wait_for_peers(min_peers=int(config['MODELS']['num_of_exits']), 
                                        check_interval=int(config['P2P']['wait_for_peers_interval']))

        asyncio.create_task(p2p_handler.subscribe_to_messages())
        #await p2p_handler.start_listening()

        role = config['ROLE']['role']
        timeout = int(config['P2P']['infernece_timeout'])

        #p2p_handler.local_peer_status = PeerStatus(p2p_handler.local_peer_id, Status.READY, role)
        for _ in range(ATTEMPTS):
            await p2p_handler.publish_status(p2p_handler.local_peer_status)
        await asyncio.sleep(10)
        p2p_handler.local_peer_status.status = Status.READY
        p2p_handler.network_state.update_peer_status(p2p_handler.local_peer_status)

        for _ in range(ATTEMPTS):
            await p2p_handler.publish_status(p2p_handler.local_peer_status)
        await asyncio.sleep(10)

        #await p2p_handler.publish_status(p2p_handler.local_peer_status)

        if role == 'edge':
            await perform_edge_inference(p2p_handler, timeout, device)
        elif role == 'cloud1':
            await perform_cloud1_inference(p2p_handler, config, device)
        elif role == 'cloud2':
            await perform_cloud2_inference(p2p_handler, config, device)

    except Exception as e:
        logging.exception(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())