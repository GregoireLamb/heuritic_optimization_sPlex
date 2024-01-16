# Heuristic Optimization for the Weighted s-Plex Editing Problem

Programming exercise for the course [192.137 Heuristic Optimization Techniques (VU 3,0)](https://tiss.tuwien.ac.at/course/educationDetails.xhtml?dswid=7017&dsrid=945&semester=2023W&courseNr=192137) at TU Wien 2023W.

**Group 05**: Joan Salvà Soler, Grégoire de Lambertye

## Problem description
In this project, we implement heuristic methods for the s-Plex Editing Problem. 
We consider the Weighted s-Plex Editing Problem (WsPEP), which is a generalization of the s-Plex
Editing Problem which in itself is a generalization of the Graph Editing Problem. We are given an
undirected graph ``G = (V, E)``, a positive integer value s, and a symmetric weight matrix W which assigns
every (unordered) vertex pair ``(i, j) : i, j ∈ V, i ̸= j`` a non-negative integer weight ``w_ij``.
An s-Plex of a graph is a subset of vertices ``S ⊆ V`` such that each vertex has degree at least ``|S| − s`` and
there exist no edges to vertices outside the s-Plex, i.e., ``i, j ∈ V, i ∈ S, j /∈ S =⇒ (i, j) ∈/ E``. Note
that a clique (complete graph on a vertex subset) is a 1-plex. The goal is to edit the edges of the graph by deleting 
existing edges and/or inserting new edges such
that the edited graph consist only of non-overlapping s-Plexes and such that the sum over all weights of
the edited edges is minimal.
Let a candidate solution be represented by variables xij ∈ {0, 1}, i, j ∈ V, i < j, where a value 1 indicates
that edge ``(i, j)`` is either inserted (if ``(i, j) ̸∈ E``) or deleted (if ``(i, j) ∈ E``) and a value 0 indicates that the
edge is not edited. The objective function is then given by
``\min f(x) = \sum_{(i,j)∈E} w_{ij} x_{ij}``

## Implemented methods
We implemented the following methods:
- **Construction Heuristics**: Greedy and Randomized Greedy construction heuristics
- **GRASP**: Greedy Randomized Adaptive Search Procedure on top of the Randomized Greedy construction heuristic
- **Local Search**: Iterated Local Search with the construction heuristic as initial solution. The neighborhood structures considered are
  - **Swap nodes**: Swap two nodes in the solution
  - **k-flips**: Remove k edges in the solution
  - **Move nodes**: Move nodes from one s-Plex to another
- **VND**: Variable Neighborhood Descent
- **Simulated Annealing**: Simulated Annealing with the construction heuristic as initial solution.
- **BRKGA**: Biased Random Key Genetic Algorithm
- **ACO**: Ant Colony Optimization

The project includes a hyperparameter optimization module that utilizes [SMAC](https://automl.github.io/SMAC3/v2.0.2/), a Bayesian Optimization framework for Black-Box optimization and algorithm configuration.

## Usage
The code is written in Python 3.8. For the installation of the required packages, run
```python
pip install -r requirements.txt
```

The main scripts are ``main.py`` and ``benchmarking.py``. The first one is used to run the algorithms on a single instance, 
while the second one is used to run the algorithms on a set of instances and writes the results to a CSV file. 

The ``documentation`` folder contains the documentation of the code. The ``instances`` folder contains the assignment 
description and problem information. The ``instances`` folder contains the instances of the problem. There are test,
tuning, and competition instances. The ``results`` folder contains the results of the algorithms on the test and tuning
instances.
