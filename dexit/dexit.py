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

from dexit.utils.state import Status, InferenceRequest, InferenceResult
from dexit.network.handler import P2PHandler
from dexit.data.dataloaders import CIFARDataLoader
#from dexit.early_exit.early_exit import * 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

### For debug purposes
ATTEMPTS = 3
torch.autograd.set_detect_anomaly(True)
# Ensure the metrics directory exists
PLOTS_DIR = '../plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

def generate_plots(accuracy, exit_counts, class_correct, class_total):
    exit_distribution_filename = f'{PLOTS_DIR}/exit_distribution.png'
    class_accuracy_filename = f'{PLOTS_DIR}/class_accuracy.png'
    cumulative_accuracy_filename = f'{PLOTS_DIR}/cumulative_accuracy.png'

    # Plot 1: Exit Distribution
    plt.figure(figsize=(10, 5))
    plt.bar(exit_counts.keys(), exit_counts.values())
    plt.title('Distribution of Exits')
    plt.xlabel('Exit Point')
    plt.ylabel('Number of Samples')
    plt.savefig(exit_distribution_filename)
    plt.close()

    # Plot 2: Accuracy by Class
    class_accuracy = {cls: class_correct[cls] / class_total[cls] for cls in class_total.keys()}
    plt.figure(figsize=(12, 6))
    plt.bar(class_accuracy.keys(), class_accuracy.values())
    plt.title('Accuracy by Class')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(class_accuracy_filename)
    plt.close()

    # Plot 3: Exit Point vs Accuracy
    total_samples = sum(exit_counts.values())
    cumulative_samples = 0
    cumulative_accuracy = []
    for exit_point, count in exit_counts.items():
        cumulative_samples += count
        cumulative_accuracy.append(accuracy * (cumulative_samples / total_samples))

    plt.figure(figsize=(10, 5))
    plt.plot(list(exit_counts.keys()), cumulative_accuracy, marker='o')
    plt.title('Cumulative Accuracy vs Exit Point')
    plt.xlabel('Exit Point')
    plt.ylabel('Cumulative Accuracy')
    plt.ylim(0, 1)
    plt.savefig(cumulative_accuracy_filename)
    plt.close()

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

async def perform_edge_inference(p2p_handler, timeout, device, num_samples=None):
    dataloader = CIFARDataLoader(num_samples=num_samples).get_test_loader()
    correct_samples = 0
    total_samples = 0
    exit_counts = {'edge': 0, 'cloud1': 0, 'cloud2': 0}
    class_correct = {i: 0 for i in range(10)}
    class_total = {i: 0 for i in range(10)}

    for data, labels in dataloader:
        batch = data.to(device)
        labels = labels.to(device)
        
        for i in range(batch.shape[0]):
            total_samples += 1
            sample = batch[i]
            
            output, exit_taken = p2p_handler.models['edge_device'](sample)
            
            if exit_taken:
                exit_counts['edge'] += 1
                exit_point = 'edge'
            else:
                cloud1_peer_id = p2p_handler.network_state.get_peer_by_role('cloud1')
                if cloud1_peer_id:
                    inference_request = InferenceRequest(peer_id=p2p_handler.local_peer_id, sample=output)
                    await p2p_handler.send_inference_request(cloud1_peer_id, inference_request)
                    
                    result = await p2p_handler.wait_for_inference_result(peer_id=p2p_handler.local_peer_id, timeout=timeout)
                    if result:
                        output = result.result
                        exit_point = result.exit_point
                        exit_counts[exit_point] += 1
                    else:
                        logging.warning("Cloud1 inference timeout")
                        exit_point = 'timeout'
                else:
                    logging.warning("Cloud1 peer not found")
                    exit_point = 'error'
            
            prediction = torch.argmax(output)
            correct = (prediction == labels[i]).item()
            class_correct[labels[i].item()] += correct
            class_total[labels[i].item()] += 1
            
            if correct:
                correct_samples += 1
            
            logging.info(f"Sample {total_samples}: Prediction: {prediction.item()}, Actual: {labels[i].item()}, Correct: {correct}, Exit: {exit_point}")
    
    accuracy = correct_samples / total_samples
    logging.info(f"Final accuracy: {accuracy:.4f}")
    logging.info(f"Exit counts: {exit_counts}")
    
    return accuracy, exit_counts, class_correct, class_total

async def perform_cloud1_inference(p2p_handler, config, device):
    while True:
        inference_request = await p2p_handler.wait_for_inference_request(timeout=60)
        if inference_request:
            sample = inference_request.sample.to(device)
            output, exit_taken = p2p_handler.models['cloud1'](sample)
            
            if exit_taken:
                exit_point = 'cloud1'
            else:
                exit_point = 'cloud2'
                cloud2_peer_id = p2p_handler.network_state.get_peer_by_role('cloud2')
                if cloud2_peer_id:
                    new_request = InferenceRequest(peer_id=inference_request.peer_id, sample=output)
                    await p2p_handler.send_inference_request(cloud2_peer_id, new_request)
                    logging.info("Request forwarded to Cloud2")
                    continue  # Don't send result back to edge yet
                else:
                    logging.warning("Cloud2 peer not found")
            
            result = InferenceResult(peer_id=inference_request.peer_id, result=output, exit_point=exit_point)
            await p2p_handler.send_inference_result(inference_request.peer_id, result)
        else:
            await asyncio.sleep(0.1)

async def perform_cloud2_inference(p2p_handler, config, device):
    while True:
        inference_request = await p2p_handler.wait_for_inference_request(timeout=60)
        if inference_request:
            sample = inference_request.sample.to(device)
            output, _ = p2p_handler.models['cloud2'](sample)
            
            result = InferenceResult(peer_id=inference_request.peer_id, result=output, exit_point='cloud2')
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

        role = config['ROLE']['role']
        timeout = int(config['P2P']['infernece_timeout'])
        num_test_samples = int(config['DATASET']['num_samples'])

        for _ in range(ATTEMPTS):
            await p2p_handler.publish_status(p2p_handler.local_peer_status)
        await asyncio.sleep(10)
        p2p_handler.local_peer_status.status = Status.READY
        p2p_handler.network_state.update_peer_status(p2p_handler.local_peer_status)

        for _ in range(ATTEMPTS):
            await p2p_handler.publish_status(p2p_handler.local_peer_status)
        await asyncio.sleep(10)


        if role == 'edge':
            accuracy, exit_counts, class_correct, class_total = await perform_edge_inference(p2p_handler, timeout, device, num_samples=num_test_samples)
            generate_plots(accuracy, exit_counts, class_correct, class_total)
            #await perform_edge_inference(p2p_handler, timeout, device, num_samples=num_test_samples)
        elif role == 'cloud1':
            await perform_cloud1_inference(p2p_handler, config, device)
        elif role == 'cloud2':
            await perform_cloud2_inference(p2p_handler, config, device)

    except Exception as e:
        logging.exception(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())