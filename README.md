# Credit Approval Prediction Using Genetic Algorithms

This repository contains a Python implementation of a genetic algorithm designed to optimize weight vectors for predicting credit approval. The project utilizes genetic algorithms to minimize the error between predicted and actual credit approvals, exploring how evolutionary computation can be applied to solve optimization problems in binary classification tasks.

## Project Structure

- `CreditCard.csv`: The dataset containing credit application data.
- `genetic_algorithm.py`: The main Python script implementing the genetic algorithm.

## Features

- **Data Preprocessing**: Converts categorical data into binary values suitable for model input.
- **Genetic Algorithm Implementation**: Includes initialization, selection, crossover, mutation, and fitness evaluation to evolve the population towards the best solution.
- **Visualization**: Generates plots to visualize the error rate across generations, helping to analyze the performance and convergence of the algorithm.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:

```bash
pip install numpy pandas matplotlib
```

### Installing

A step-by-step series of examples that tell you how to get a development environment running:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-approval-ga.git
   ```
2. Navigate to the cloned repository:
   ```bash
   cd credit-approval-ga
   ```

3. Run the script:
   ```bash
   python genetic_algorithm.py
   ```

## Usage

The script can be executed with Python 3.x, and it will read the provided `CreditCard.csv`, process the data, and run the genetic algorithm to find the optimal weights for credit approval prediction. The results will be displayed in the console and as a plot showing the evolution of the error rate across generations.

## Authors

- **Natalie Hill**
