import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in data
df = pd.read_csv("CreditCard.csv")

# Data preprocessing
df["Gender"] = df["Gender"].map({"M": 1, "F": 0})
df["CarOwner"] = df["CarOwner"].map({"Y": 1, "N": 0})
df["PropertyOwner"] = df["PropertyOwner"].map({"Y": 1, "N": 0})

# Set attributes and target
X = df[['Gender', 'CarOwner', 'PropertyOwner', '#Children', 'WorkPhone', 'Email_ID']].values
y = df['CreditApprove'].values

# Error function
def error_function(w):
    predictions = np.where(np.dot(X, w) >= 0, 1, 0)
    return np.sum((predictions - y) ** 2)

# Genetic algorithm parameters
population_size = 10
num_genes = 6
num_generations = 50
mutation_rate = 0.01

# Initialize popoulation
population = np.random.choice([-1, 1], size=(population_size, num_genes))
fitness_history = []

for generation in range(num_generations):
    # Calculate fitness
    fitness = np.array([np.exp(-error_function(individual)) for individual in population])

    # Selection
    selection_prob = fitness / fitness.sum()
    selected_indices = np.random.choice(population_size, size=population_size, p=selection_prob)
    selected_population = population[selected_indices]

    # Crossover
    offspring = []
    for i in range(0, population_size, 2):
        parent1, parent2 = selected_population[i], selected_population[i+1]
        crossover_point = num_genes // 2
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        offspring.extend([child1, child2])
    offspring = np.array(offspring)

    # Mutation
    mutation_mask = np.random.rand(population_size, num_genes) < mutation_rate
    offspring = np.where(mutation_mask, -offspring, offspring)

    # Update popoulation
    population = offspring

    # Best fitness
    best_fitness = np.max(fitness)
    best_w = population[np.argmax(fitness)]
    best_error = error_function(best_w)
    fitness_history.append(best_error)
    print(f"Generation {generation}: Best Error = {best_error}")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(fitness_history, marker='o')
plt.xlabel('Generation')
plt.ylabel('Error Rate')
plt.title('Error Rate Across Generations')
plt.show()

# Display the best solution
print("Optimal weights:", best_w)
print("Minimum error rate:", best_error)