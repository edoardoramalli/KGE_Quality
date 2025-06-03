# KGE_Quality

This repository contain the source and the result of the experiments for the paper "A Quality-aware Node Ablation Study for Link Prediction in
Knowledge Graphs" authored by E. Ramalli, C.A. Bono, C. Sancricca, C. Cappiello, M. Comuzzi, M. Vitali and B. Pernici.

## Repository Structure

```text
├── Ablation
├── Data_Collection
├── Development
├── Energy Models
├── GraphMeasure
├── Measure
├── Plotting
├── README.md
├── Results
├── Training
└── __init__.py

```
### `Data_Collection` 
The main scripts are:
- `downloadDatasets.py` routine downloads datasets pre-divided in training, test and validation test.
- `margeDatasets.py` merges the pre-defined training, test and validation splits into a single dataset.
- `createSplits.py` splits the datasets 5 times in training, test and validation set, according to a random seed.

The folder `Datasets_Complete` contains the full datasets and the splits used for this work.

### `Measure`
The script `measureKG.py` computes the entities centrality metrics.

### `Ablation`
The script `createTask.py` creates the ablated version of each dataset/split according to a certain criteria (quantity and
selection of entities). Each ablated dataset will be provide as input (a.k.a. a task) as training/test set to each KGE
model considered in the study.

### `Training`
The script `consumeTask.py` consume a task, i.e., take as input an ablated dataset and a KGE model and train it, recording
the energy consumption. The hyperparameters are stored in `config.json`, including the training random seed.

### `Results`