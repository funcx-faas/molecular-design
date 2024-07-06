# ML-in-the-loop molecular design with Globus Compute

This tutorial is based on a [tutorial for Parsl](https://github.com/ExaWorks/molecular-design-parsl-demo))

This repository contains a tutorial showing how Globus Compute can be used to write a machine-learning-guided search for high-performing molecules.

The objective of this application is to identify which molecules have the largest ionization energies (IE, the amount of energy required to remove an electron). 

IE can be computed using various simulation packages (here we use [xTB](https://xtb-docs.readthedocs.io/en/latest/contents.html)); however, execution of these simulations is expensive, and thus, given a finite compute budget, we must carefully select which molecules to explore. 

In this example, we use machine learning to predict molecules with high IE based on previous computations (a process often called [active learning](https://pubs.acs.org/doi/abs/10.1021/acs.chemmater.0c00768)). We iteratively retrain the machine learning model to improve the accuracy of predictions. 

## Installation

The demo uses a few codes that are easiest to install with Anaconda. Our environment should work on both Linux and OS X can can be installed by:

```bash
conda env create --file environment.yml
```

