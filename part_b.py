import random
import matplotlib.pyplot as plt

# Constants for our experiment - problem exhibits a fast learning dropoff
# so could reduce generations?
MAX_WEIGHT = 35
POPULATION_SIZE = 50
GENERATIONS = 50
MUTATION_RATE = 0.1

lorry_weights = [3, 8, 2, 9, 7, 1, 8, 13, 10, 9]
lorry_values = [126, 154, 256, 526, 388, 245, 210, 442, 671, 348]

def fitness(chromosome):
    weight = sum(g * weight for g, weight in zip(chromosome, lorry_weights))
    value = sum(g * value for g, value in zip(chromosome, lorry_values))

    if weight > MAX_WEIGHT:
        return 0

    return value

# Take a sample size of 3, evaluate their fitness, take the winner
def tournament_selection(population, tournament_size=3):
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=fitness)

# Crossover between the winning members of selection process
# Pick random crossover point, then take half from each side at that index and return
def single_point_crossover(parent_one, parent_two):
    chromosome_crossover_point = random.randint(1, len(parent_one) - 1)
    print(f'Crossover occured at index {chromosome_crossover_point}')
    return parent_one[:chromosome_crossover_point] + parent_two[chromosome_crossover_point:]

# We mutate only if our random generated value is greater than our mutation rate
def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        allele = random.randint(0, len(chromosome) - 1)
        chromosome[allele] = (1 - chromosome[allele])
    return chromosome


# Pass the best, average and worst fitness scores per generation to generate a 
# matplotlib graph over the runtime of the GA
def plot_fitness_scores(best, avg, worst):

    plot_generations = [g for g in range(GENERATIONS)]

    plt.plot(plot_generations, best, label='Best Fitness', color='r', linestyle='--')
    plt.plot(plot_generations, avg, label='Average Fitness', color='g', linestyle='--')
    plt.plot(plot_generations, worst, label='Worst Fitness', color='b', linestyle='--')

    plt.title('Best Fitness Score Per Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.legend()
    plt.grid(True)
    plt.savefig("part_b_plot.png")

    print(f'Saved plot to working directory - part_b_plot.png')



def ga():
    best_fitness_scores = []
    avg_fitness_scores = []
    worst_fitness_scores = []

    population = [[random.randint(0, 1) for _ in range(len(lorry_weights))] for _ in range(POPULATION_SIZE)]
    
    for _ in range(GENERATIONS):
        population = sorted(population, key=fitness, reverse=True)[:POPULATION_SIZE]
        
        best_fitness_scores.append(fitness(population[0])) 
        avg_fitness_scores.append(sum(fitness(chromosome) for chromosome in population) / len(population))  # sum avg fitness
        worst_fitness_scores.append(fitness(population[-1])) 
        
        new_population = population[:10]
        
        for _ in range(POPULATION_SIZE):
            parent_one = tournament_selection(population)
            parent_two = tournament_selection(population)
            child = mutate(single_point_crossover(parent_one, parent_two))
            new_population.append(child)

        population = new_population
    
    best_solution = max(population, key=fitness)
    total_weight = sum(g * weight for g, weight in zip(best_solution, lorry_weights))
    total_value = fitness(best_solution)
    
    plot_fitness_scores(best_fitness_scores, avg_fitness_scores, worst_fitness_scores)
    print(f"GA's best chromosome output is {best_solution}, total weight is {total_weight}, total value is {total_value}")


if __name__ == "__main__":
    ga()