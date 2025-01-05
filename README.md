# Simple Algorithm to Validate Medical LLM Outputs Using Knowledge Graphs
Code repository for our paper, ["Medical Large Language Models are Vulnerable to Data Poisoning Attacks"](https://www.nature.com/articles/s41591-024-03445-1) (Nature Medicine, 2024).

1. Install miniconda (https://docs.anaconda.com/free/miniconda/)
2. Create a conda environment: `conda create -n defense-algorithm python=3.11`
3. Activate the environment: `conda activate defense-algorithm`
4. Change to this directory: `cd <path/to/this/dir>`
5. Install requirements using pip: `pip install -r requirements.txt`
6. Run the script using the toy dataset: `python screen_outputs.py`

Note: Although our code is released under an MIT license, the implemented embedding models possess their own licensing agreements.
