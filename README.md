# Relational Pooling for Graph Representations

## Overview
This is the code associated with the paper [Relational Pooling for Graph Representations](https://arxiv.org/abs/1903.02541).
Accepted at ICML, 2019.

Our first task evaluates RP-GIN, a powerful model we propose to make Graph Isomorphism Network (GIN) [Xu et. al. 2019](https://arxiv.org/abs/1810.00826) more powerful than its corresponding WL[1] test. 
Our second set of tasks uses molecule datasets to evaluate different instantiations of RP.

The models are described in plain English in the appendix of our paper, but feel free to contact us with any questions (see below).  

## Requirements
* [PyTorch](https://www.pytorch.org)
* Python 3

For the first set of tasks, you will need
* SciPy
* scikit-learn
* docopt and schema for parsing arguments from command line

For the molecular tasks, you will need
* [DeepChem](https://github.com/deepchem/deepchem) and its associated dependencies

## Examples: How to Run
* An example call for the synthetic tasks follows.  We trained these models on CPUs.  Please see the docstring for further details
```
python Run_Gin_Experiment.py --cv-fold 0 --model-type rpGin --out-weight-dir /some/path --out-log-dir /another/path/ --onehot-id-dim 10
```
* Now we show examples for the molecular tasks.  The Tox 21 dataset is smaller so we demonstrate with that.
For the molecular k-ary tasks:
```
python Duvenaud-kary.py 'tox_21' 20
```
* For the molecular RP-Duvenaud tasks:
```
python rp_duvenaud.py 'tox_21' 'unique_local' 0
```
* For the molecular RNN task:
```
python RNN-DFS.py 'tox_21'
```

## Data
* The datasets for the first set of tasks are available in the Synthetic_Data directory.
* The datasets for the molecular tasks are all available in the DeepChem package.

## Questions and Contact
Please feel free to reach out to Ryan Murphy (murph213@purdue.edu) if you have any questions.

## Citation
If you use this code, please consider citing our paper.  Bibtex entries for all ICML papers can be found on the [Proceedings of Machine Learning Research](http://proceedings.mlr.press/) website.  