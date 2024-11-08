import random as rd

def print_fpop(f_pop):
    for indexp in f_pop:
        print(indexp)


def i_pop(size, chromosome):
    pop = []
    for inx in range(size):
        pop.append(rd.choices(range(2), k=chromosome))
    return pop


def fitness_f(individual):
    count_B = individual[:8].count(1) + individual[24:].count(1)
    count_G = individual[8:24].count(0)
    
    return count_G + count_B

def Roulette_wheel(pop, fitness):
    parents = []
    fitotal = sum(fitness)
    normalized = [x / fitotal for x in fitness]
    f_cumulative = []
    index = 0

    for n_value in normalized:
        index += n_value
        f_cumulative.append(index)

    pop_size = len(pop)

    for index2 in range(pop_size):
        rand_n = rd.uniform(0, 1)
        individual_n = 0
        for fitvalue in f_cumulative:
            if rand_n <= fitvalue:
                parents.append(pop[individual_n])
                break
            individual_n += 1
    return parents


def mutate(chromo):
    for idx in range(len(chromo)):
        if rd.random() < 0.08:
            chromo = chromo[:idx] + [1 - chromo[idx]] + chromo[idx + 1 :]
    return chromo


def mating_crossover(parent_a, parent_b):
    offspring = []
    cut_point = rd.randint(1, len(parent_a) - 1)

    offspring.append(parent_a[:cut_point] + parent_b[cut_point:])
    offspring.append(parent_b[:cut_point] + parent_a[cut_point:])

    return offspring


def main():
    generations = 100
    size = 50
    chromosome = 32

    population = i_pop(size, chromosome)

    for generation in range(generations):
        print("Current generation: ", generation)

        fitness_arr = list(map(fitness_f, population))
        sorted_fitness = sorted(fitness_arr)
        if sorted_fitness[-1] >= 30:
            print("A good solution found with a fitness score of", sorted_fitness[-1], "in", generation,"generations")
            print("************")
            print(population, '\n', fitness_arr)
            break
        
        parents = Roulette_wheel(population, fitness_arr)
        new_population = []

        while len(new_population) < size:
            parent_a, parent_b = rd.sample(parents, 2)
            if rd.random() < 0.75:
                offspring = mating_crossover(parent_a, parent_b)
                new_population.extend([mutate(child) for child in offspring])
            else:
                new_population.extend((parent_a, parent_b))

        population = new_population
        print(list(map(fitness_f, population)))
        
    print("\n\n\nMax generations reached (or good solution found), here is the population of the final generation")
    print("========================")
    print(population)
    print("========================")
    print("individuals' fitness scores sorted ascending order")
    print(sorted(list(map(fitness_f, population))))

main()
# Ideal solution [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
