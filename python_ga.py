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
    count = 0
    for i in range(len(individual)):
        if i < 8 or i >= 24:
            if individual[i] == 1:
                count += 1
        else:
            if individual[i] == 0:
                count += 1
    return count

def Roulette_wheel(pop, fitness):
    parents = []
    fitotal = sum(fitness)
    normalized = [x / fitotal for x in fitness]

    """ print('normalized fitness')
    print('________________________')
    print_fpop(normalized)
    print('________________________') """

    f_cumulative = []
    index = 0

    for n_value in normalized:
        index += n_value
        f_cumulative.append(index)

    pop_size = len(pop)
    """ print('cumulative fitness')
    print('________________________')
    print_fpop(f_cumulative)
    print('________________________') """

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
        if rd.random() < 0.1:
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
    size = 10
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


main()
# Ideal solution [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,]
