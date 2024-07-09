======
DEXIT
======

Decentralized Early Exit Inference Tool
=======================================

DEXIT is a distributed inference system that implements early exit strategies across multiple nodes. It allows for efficient inference by potentially terminating the process early at different stages of the network, distributed across edge and cloud devices.

Features
--------

- Distributed inference across edge and cloud nodes
- Early exit strategy for efficient processing
- Peer-to-peer communication using libp2p
- Support for multiple models (edge, cloud1, cloud2)
- Asynchronous processing and communication

Components
----------

1. Edge Device: Initiates the inference process and handles early exits.
2. Cloud1: Intermediate processing node with early exit capability.
3. Cloud2: Final processing node for complex inferences.

Requirements
------------

- Python 3.7+
- PyTorch
- asyncio
- tqdm
- git-lfs

Installation
------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/anaskalt/dexit.git
      cd dexit

2. Install the required packages:

   .. code-block:: bash

      pip install -r requirements.txt

3. Install git-lfs:

   .. code-block:: bash

      sudo apt install git-lfs

4. Pull models:

   .. code-block:: bash

      git lfs pull

Configuration
-------------

1. Update the `conf/node.conf` file with appropriate settings for each node (edge, cloud1, cloud2).
2. Ensure the model paths in the configuration file point to your pre-trained models.

Usage
-----

To run a node (edge, cloud1, or cloud2):

.. code-block:: bash

   python dexit.py

Make sure to run the script on different machines or in different environments for each node type.

Project Structure
-----------------

- `dexit.py`: Main script for running the distributed inference
- `network/handler.py`: Handles P2P network operations
- `utils/state.py`: Defines data structures for network state and inference objects
- `data/dataloaders.py`: Provides data loading functionality
- `early_exit/`: Contains the early exit model implementations
- `conf/node.conf`: Configuration file for node settings

Contributing
------------

Contributions to DEXIT are welcome! Please feel free to submit a Pull Request.

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.