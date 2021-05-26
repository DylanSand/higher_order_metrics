# higher_order_metrics
A preliminary tool for measuring some higher-order graph metrics in the context of graph neural networks (GNNs).

## Requirements
- Pytorch
- Torch Sparse
- Pytorch Geometric

## Overview
This repo can be used to measure and compare some prototype higher-order metrics on graphs for use in graph neural networks.<br>
It can also be used to split up the [Varmisuse](https://arxiv.org/pdf/1711.00740.pdf) dataset and run individual experiments on each Github repo within Varmisuse without bundling them all together.

This tool's purpose is to probe some potential candidates for future graph metrics that may be especially useful for determining whether a particular graph dataset would benefit from the use of higher-order graph models rather than simpler message-passing models that are [less powerful](https://arxiv.org/pdf/1810.00826.pdf).
The metrics in question should ideally be scalable and intuitively explainable while correlating fittingly with higher-order model performance.

## Metrics
There are two major metrics currently captured in this tool:

### **Cyclic Probability:**
This metric corresponds to the average probability of stepping through a cycle on the graph during a non-backtracking random walk of length *n*.
The reason this metric is important is that it reflects the density of the cycles in the graph. For large *n*, a value of this metric close to 1 indicates that the underlying graph is so cycle-dense that it is likely a cycle already. Meanwhile a value close to 0 indicates the graph is almost just a tree.
This is relevant because message-passing GNNs represent the features of the nodes in a tree-like pattern, so we expect tree-like graphs to perform equally well when using simple and higher-order models.

**Calculation:** The calculation of this metric can be done by calculating a variation of a standard transition probability matrix on the directed edges of the graph rather than the nodes and additionally adding the condition that the tail of an edge in a path cannot be equal to the head of the following edge. This process is outlined in [this paper](https://arxiv.org/pdf/1603.05553.pdf) that explains how such a transition matrix is constructed. This matrix can then be used in conjunction with an "edge adjacency" matrix to calculate the transition probability matrix for a graph with paths of any desired length *i*. The diagonals of *i*-transition matrices are then combined for all *i* in [1, *n*] to get the final probabilities of returning to any starting node when doing a random walk of length *n*. This final value is this metric we use.

This calculation can be modified further to instead return the analagous stable dsitribution of the edge transition probability matrix in a process outlined in the paper linked above.

### **Fully-Connected (FC) Layer Comparison:**
This metric is not as rigorously defined as the previous one. We simply add a FC layer to the end of the model and see how that affects performance over multiple runs. The FC layer can also replace an existing final layer in the analysis. Making this change explores how ignoring the graph structure of the data (or rather, how turning the graph into a complete graph) can be used to uncover longer-range dependencies between the nodes. If performance improves when the FC layer is added, it may indicate what GNN models work best with the dataset.

This experiment can also be run multiple times using different GNN models for the pre-FC layers to compare which ones bring out the most striking difference in performance between the base and FC-augmented models.

## Why Varmisuse?
The reason this tool emphasizes analysis on the Varmisuses dataset is because of the dataset's predisposition to long-range dependencies between nodes. Varmisuse is a real-world dataset made up of code from Github repositories that have been deconstructed to make program graphs that represent the relationships between tokens and variables in the code. These graphs have the somewhat rare quality of containing a fair number of nodes that are far apart in the space of the graph but dependant on each other nonetheless (one can imagine how two tokens from different parts of a code repo can refer to the same variable).

This means that the Varmisuse dataset can appropriately be use as a benchmark for metrics attempting to capture this long-range quality within graph datasets.

## Running basic experiments
If you have installed the requirements listed above, all of the model experiments should work except for the MixHop model. To get this last model-type working, please clone the [MixHop](https://github.com/benedekrozemberczki/MixHop-and-N-GCN) repo into the root folder of this repo.

To run non-Varmisuse experiments, simply run the *main.py* file using the *input_path* flag in the following way:<br>
`python main.py --input_path <PATH TO higher_order_metrics REPO>`

You can change many configurations about the experiments by checking the *CONFIG.py* file and looking at options for each of the other Python files using the sections listed there.

## Running an experiment on a single repo of Varmisuse
1. Install the [ptgnn](https://github.com/microsoft/ptgnn) library and clone the repo into the root folder of this repo.
2. Download the [Varmisuse dataset ZIP](https://www.microsoft.com/en-us/research/publication/learning-represent-programs-graphs/) and place it in the root folder of this repo
3. Run *reorg_data.sh*
4. Change the word in *cur_dataset* to the name of a repo in Varmisuse you want to run an experiment on (examples include: **ninject**, **choco**, **dapper**, etc.)
5. Run *run_varmisuse.sh*

Every subsequent experiment only requires you to rerun steps 4 and 5.

## Future directions and goals
As previously stated, the goal of this project is to formulate useful and scalable metrics that can be used to do higher-order analysis on graph datasets. These kinds of metrics could in turn inform future GNN researchers of the validity of their claims when testing and evaluating the effectiveness of potential higher-order graph models.
