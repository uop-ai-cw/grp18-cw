import random

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

def tournament_selection(population, tournament_size=3):
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=fitness)

def single_point_crossover(parent_one, parent_two):
    chromosone_crossover_point = random.randint(1, len(parent_one) - 1)
    return parent_one[:chromosone_crossover_point] + parent_two[chromosone_crossover_point:]

def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        allele = random.randint(0, len(chromosome) - 1)
        chromosome[allele] = (1 - chromosome[allele])
    return chromosome




def ga():
    population = [[random.randint(0, 1) for _ in range(len(lorry_weights))] for _ in range(POPULATION_SIZE)]
    
    for _ in range(GENERATIONS):
        population = sorted(population, key=fitness, reverse=True)[:POPULATION_SIZE]
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
    
    print(F"GA's best chromosone output is {best_solution}, total weight is {total_weight}, total value is {total_value}")


if __name__ == "__main__":
    ga()

# words from ioannis

# display its working???
# how do we know its working correctly for every step. does it solve the problem 
# optimise as much as possible
# end generations if a solution we want 