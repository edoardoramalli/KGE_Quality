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






### `Training`


### `Results`