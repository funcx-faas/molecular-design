# ML-in-the-loop molecular design with Globus Compute

This repository contains a tutorial showing how Globus Compute can be used to write a machine-learning-guided search for high-performing molecules.

The objective of this application is to identify which molecules have the largest ionization energies (IE, the amount of energy required to remove an electron). 

IE can be computed using various simulation packages (here we use [xTB](https://xtb-docs.readthedocs.io/en/latest/contents.html)); however, execution of these simulations is expensive, and thus, given a finite compute budget, we must carefully select which molecules to explore. 

In this example, we use machine learning to predict molecules with high IE based on previous computations (a process often called [active learning](https://pubs.acs.org/doi/abs/10.1021/acs.chemmater.0c00768)). We iteratively retrain the machine learning model to improve the accuracy of predictions. 

This tutorial is based on a [tutorial for Parsl](https://github.com/ExaWorks/molecular-design-parsl-demo))

## Installation

The demo builds on several packages to compute molecular properties and to build the machine learning loop. These dependencies can be easily deployed with Conda or using Docker as shown below. 

```bash
conda env create --file environment.yml
```

```bash
docker build -t moldesign . 
docker run -it moldesign /bin/bash 
```

## Globus Compute Endpoint

The demo requires deployment of a Globus Compute endpoint. Importantly, the same dependencies must be available in the endpoint's environment. You can use the same Conda or Docker environment as above.  

First, configure your Globus Compute endpoint (note, you must be in the conda environment)

```bash
conda activate moldesign
globus-compute-endpoint configure
```

Second, start your endpoint and authenticate via Globus to securely pair your endpoint with your account. Optionally, you may update the endpoint configuration to use additional cores or make use of HPC resources. See the [Globus Compute documentation](https://globus-compute.readthedocs.io/en/latest/endpoints/single_user.html) for details on configuring endpoints. 

```bash
globus-compute-endpoint start default
```

Make note of the endpoint's UUID to add to your notebook. 


