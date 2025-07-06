# Incremental Unlearning vs. Retraining Analysis Suite
This repository provides a testing suite to analyze the performance and fidelity of incremental machine learning unlearning methods compared to the "gold standard" of retraining a model from scratch. It uses a Dockerized environment to ensure complete reproducibility.

## Features
Algorithm Implementation: Includes a Python implementation of an incremental ForgettableERM class.

*Simulation Suite:* A testing script to run multiple simulations, comparing incremental updates against full retraining.

*Statistical Analysis:* Automatically calculates confidence intervals for the error between the methods.

*Reproducible Environment:* Configured with a Dockerfile and devcontainer.json for one-click setup in GitHub Codespaces or local Docker.

## Repository Structure
```
.
├── .devcontainer/
│   ├── devcontainer.json
│   └── Dockerfile
├── src/
│   ├── memory_pair.py
│   └── simulation.py
├── data/  
├── results/ 
├── .gitignore
├── requirements.txt 
└── README.md
```

## Getting Started
You can run this project in two ways: using GitHub Codespaces (recommended for ease of use) or by building the Docker container locally.

### *Option 1:* GitHub Codespaces (Recommended)

Navigate to the main page of this repository on GitHub.

Click the green `< > Code` button.

Select the Codespaces tab and click Create codespace on main.

GitHub will automatically build the containerized environment and open it in a web-based VS Code editor. No local setup is required.

### *Option 2:* Local Docker Environment

Prerequisites: 
You must have Docker installed on your local machine.

Clone the repository:

```bash
git clone https://github.com/kennonstewart/unlearning-simulations
cd unlearning-simulations
```

Build the Docker image:

```bash
docker build -t unlearning-exp -f .devcontainer/Dockerfile .
```

Run the container interactively: This command starts the container and mounts your local project directory into the /app folder inside the container.


```bash
docker run -it --rm -v "$(pwd)":/app unlearning-exp
```

You are now inside the container's shell and ready to run the experiment.

Usage
Once your environment is running (either in Codespaces or a local Docker container), execute the main simulation script from the terminal:

```bash
python src/simulation.py
```

This will run 50 simulations comparing the incremental deletion method to a full retrain. It will then print a statistical analysis of the results.

## Example Output

```
🚀 Starting 50 simulations...
...
--- ✅ Simulation Analysis ---
Ran 50 successful simulations.

📊 ## Incremental Deletion vs. Retraining
The relative error is, on average, 14.12%
95% Confidence Interval: [13.45%, 14.79%]
```
