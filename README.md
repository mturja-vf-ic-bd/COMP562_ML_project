# Graph Classification with Multi-Scale Features
This is the final project for the Machine Learning course COMP562.001.SP19. The project is on graph classification using Graph Convolutional Network (GCN).

## Network Architecture
### Description
The building block of our network is a graph convolution layer introduced in [1]. This layer incorporates 
neighborhood information into each node to update it's feature. We have used several these GCN layers to
generate feature at different scales. Then we concatenate the output from each of these layers to build a
multi-scale representation of the node features. To go from node level feature to graph level feature we used
two different pooling mechanism -- *Max Pooling* and *Attentive Pooling*.

### Diagram
To be added

## Run the code
Dependecies:
- python3
- torch
- torch_geometric

Run: `python3 main.py`

## Code Organization
- `model.py` contains the network architecture
- `train_test.py` contains the training and testing code
- `settings.py` contains global arguments shared accross all files
- `utils.py` contains some utility functions
- `main.py` is the entry point of the code


## References
[1] T. Kipf and M. Welling. *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR 2017.