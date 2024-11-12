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

############################################# i put second solution underneath, we can combine

import random #for solution 2

bit_string = "01100111010011011101111100001111"

def fitness(bits):
    B1 = bits[:8]
    B2 = bits[24:]
    G1 = bits[16:24]
    G2 = bits[8:16]
    count_B1 = B1.count('1')
    count_B2 = B2.count('1')
    count_G1 = G1.count('1')
    count_G2 = G2.count('1')

    return (count_B1 + count_B2) - (count_G1 + count_G2) #maximize 1s in blue and minimise green

def mutate(bits):   # GA step (mutation and crossover)
    bits = list(bits)
    i = random.randint(0, len(bits) - 1)
    bits[i] = '1' if bits[i] == '0' else '0' #flip bits, if 0 then 1 if 1 then 0
    return ''.join(bits)

best_bits = bit_string       #initial bit_string as the best
best_score = fitness(bit_string) ## valuate the score of the initial bit_string

for i in range(1000):  # no of generations
    candidate = mutate(best_bits)   #mutated version of the current best configuration
    score = fitness(candidate)  #evaluate fitness of mutated
    if score > best_score: ## if better then   update the best configuration and score
        best_bits = candidate
        best_score = score
print("Original Bit String:", bit_string)
print("Optimized Bit String:", best_bits)
print("Maximized 1s in Blue, Minimized in Green:", best_score)
