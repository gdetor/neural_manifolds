# neural_manifolds
Topological and geometrical analysis of neural manifolds in sequential and interleaved learning schedules.
This repository contains all the Python scripts used in the paper:



## Organization
  - **parameters.json** Inclues all the parameters for running the Circles/Moons classification
  - **parameters_mnist.json** Contains the parameters for the FashionMNIST/MNIST classification
  - **commonm.py** Implements all the auxiliary functions used in this work from most of the scripts
  - **metrics.py** Contains all the metrics used in this work (participation ratio, sparseness, etc)
  - **toy_classifier.py** Implements the Circles/Moons classifier (both sequential and interleaved learning schedules)
  - **classic_toy_classifier.py** Implements a classic classifier that runs on either Moons or Circles dataset
  - **mnist_classifier.py** Implements the FahsioMNIST/MNIST classifier (both sequential and interleaved learning schedules)
  - **classic_mnist_classifier.py** Implements a classic classifier that runs on either MNSIT or Fashion-MNIST dataset
  - **tda_analysis.py** Performs a TDA analysis on the neural preactivations and computes the Bottleneck distances using a permutation test
  - **tda_visualization.py** Plots the Bottleneck distances between different conditions
  - **convex_hull_analysis.py** Computes the convex hull intersection points between two neural representations and their Jaccard index
  - **convex_hull_visualization.py** Visualizes the geometric measures (Jaccard, and Convex Hull intersection points)

## How to run the scripts
To run the scripts first make sure you have installed all the requirements (see at the bottom of this page).
and then you can type:
```bash
$ python3 warmup.py
```
The `warmup` script will create all the necessary folders to store the data and the results of the numerical experiments.

```bash
$ python3 toy_classifier.py
```
That would run a classifier on Circles/Moons datasets using either a sequential or an interleaved schedule. 
Similarly, for the FahsioMNIST/MNIST you can type:
```
$ python3 mnist_classifier.py
```
You can alter the parameters of the Circles/Moons and FahsionMNIST/MNIST classifications
via the files `parameters.json` and `parameters_mnist.json`, respectively.



## Tested Platform
  - Ubuntu 24.04.4 LTS (6.17.0-23-generic #23~24.04.1-Ubuntu)
  - Python 3.12.3
  - GCC 13.3.0

## Requirements
  - Torch / Torchvision
  - Numpy
  - Scipy
  - Scikit-learn
  - Gudhi
  - Multiprocess
