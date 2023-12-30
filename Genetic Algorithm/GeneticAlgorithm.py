import numpy as np
import random
import matplotlib.pyplot as plt
def fitness_function(x, y):
    return (1 - x) ** 2 * np.exp(-x**2 - (y + 1)**2) - (x - x**3 - y**3) * np.exp(-x**2 - y**2)

def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        x = random.uniform(-3, 3)
        y = random.uniform(-3, 3)
        population.append([x, y])
    return population

def crossover(parent1, parent2, crossover_rate):
    child = parent1.copy()
    if random.random() < crossover_rate:
        crossover_point = random.randint(0, len(parent1) - 1)
        child[crossover_point:] = parent2[crossover_point:]
    return child

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] += random.uniform(-0.1, 0.1)  # 임의의 작은 변화
    return individual

def genetic_algorithm(population_size, generations, crossover_rate, mutation_rate):
    population = initialize_population(population_size)
    best_solution = None
    best_fitness = float('-inf')
    best_fitness_history = []

    for generation in range(generations):
        new_population = []
        for _ in range(population_size):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = crossover(parent1, parent2, crossover_rate)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

        best_individual = max(population, key=lambda ind: fitness_function(ind[0], ind[1]))
        current_fitness = fitness_function(best_individual[0], best_individual[1])

        if current_fitness > best_fitness:
            best_solution = best_individual
            best_fitness = current_fitness

        best_fitness_history.append(current_fitness)

        print(f"Generation {generation}: Best Fitness = {current_fitness:.6f}")

    return best_solution, best_fitness_history

population_size = 6
generations = 100
crossover_rate = 0.7
mutation_rate = 0.001
best_solution, best_fitness_history = genetic_algorithm(population_size, generations, crossover_rate, mutation_rate)
x_values = list(range(generations))
plt.plot(x_values, best_fitness_history)
plt.show()
