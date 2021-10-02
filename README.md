# Samsung AI Challenge for Scientific Discovery
- Date: 2021.08.04 ~ 2021.09.27 18:00
- Task: Molecular property prediction (The gap between S1 and T1 energy)
- Result: 21st place / 220 teams

## Overview
Generating node-level feature / edge type</a>
- Using only atom type (13 dim)
- Each atom is embedded into 256 dimensional vector by a simple linear transformation
- There are 6 edge type : the number of combinations of (`bond type`, `bond stereo`)

Relational Graph Convolutional Network (RGCN)
- Total 8 RGCN layers each of which has 256 channels 
- Skip-connections 
- Using the sum of node representations followed by one hidden layer MLP as the graph representation

Readout phase
- Multi layers perceptron with 2 hidden layers (1,024 dim, 512 dim)
- Dropout with p=0.3
- Directly predicting `ST1 gap`

10-Fold ensembling
- Taking the simple average of 10 models

## Run
1. Data preparation
- `dir_data`: the directory where train.csv, dev.csv, and test.csv are stored
- `dir_output`: the directory where the preprocessed `tgm.data.Data` files will be stored.

~~~
python gnn_preprocess.py --dir_data './data' --dir_output './outputs/rgcn'
~~~

2. Training a single model and predicting test data
- Modify `TrainConfig` in `config.py`
- It gives public score about 0.127

~~~
python train.py
~~~

