# Topologic features Pair based Graph Anomaly Detection - TPGAD
A repostiory of the TPGAD algorithm, a graph-based anomaly detection algorithm applied on graphs series.

Graph-based anomaly detection methods are often based on macroscopic features of graphs, such as the total number of edges, or sub-graph frequency. However, the values of these features often vary with the edge and vertex sampling rates. We here propose a novel approach to detect human-defined anomalous graphs based on the relation between pairs of graph properties, instead of single features. The relation between such pairs is only weakly sensitive to the sampling level of either vertices or edges.
We use efficient algorithms to compute for each vertex a set of features, including topological measures, such as the degree, the clustering coefficient, and the frequency of small connected sub-graphs. We then compute the expected functional relation between pairs of features and choose maximally informative pairs of features to predict deviations from the expected relations.



## Installation
```
$ git clone https://github.com/tpgad/TPGAD.git
$ cd TPGAD
$ pip install -r requirements.txt
```

## repository hirarchy
Make the following directories a source root directories:
```
graph-ad
graph-ad/tools
graph-measures
graph-measures/features_algorithms
graph-measures/features_algorithms/accelerated_graph_features
graph-measures/features_algorithms/edges
graph-measures/features_algorithms/motif_variations
graph-measures/features_algorithms/vertices
graph-measures/features_infra
graph-measures/features_meta
graph-measures/graph_infra
graphs-package
graphs-package/features_processor
graphs-package/GRAPHS_INPUT/multi_graph
graphs-package/GRAPHS_INPUT/multi_graph/EnronInc_by_day
graphs-package/multi_graph
graphs-package/temporal_graphs
```
## General Instructions
For running TPGAD follow the instruction below:
```
1. Set the hyper-paramters in the TPGAD/graph-ad/params/X.json (X is the required dataset)
2. From TPGAD/graph-ad run anomaly_detection.py with the relevant datasetes params path
```

## Results
We show that TPGAD outperforms existing methods in multiple standard graph-based anomaly detection datasets.

|    Dataset         | Accuracy      | Recall  | Specificity  |
| -------------      |:-------------:| -----:|-----:|
| Enron Inc.         | **96.35**     |**42.86**|**97.27**|
| Reality Mining     | 62.20           |  **91.07**  |**48.52**|
| Tweitter Security  | **73.75**           |  **94.74**  |**67.21**|
| Security Repository| 80.53       |  63.95  |**82.64**|

