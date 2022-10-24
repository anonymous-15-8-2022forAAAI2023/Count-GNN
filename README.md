
# Count-GNN
We provide the implementaion of Count-GNN model on Small,Large,MUTAG, OGB-PPA dataset.

The repository is organised as follows:
- baselines/: contains baselines' models
- count-gnn/: contains our model.
- converter/: contains code to convert graphs to igraph format
- generator/: contains code to generate queries and graphs



## Package Dependencies

* tqdm
* numpy
* pandas
* scipy
* tensorboardX
* sklearn
* python-igraph
* torch >= 1.3.0
* dgl == 0.4.3post2


## Running experiments

To run _train.py:
- python _train.py --model EDGEMEAN --predict_net FilmSumPredictNet --emb_dim 4 --ppn_hidden_dim 12 --weight_decay_film 0.0001

To run evaluate.py:
- python evaluate.py ../dumps/MUTAG

