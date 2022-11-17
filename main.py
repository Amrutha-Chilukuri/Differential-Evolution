import random
import math
import numpy as np
import matplotlib.pyplot as plt


# Setting the constants
K = 0.5
Cr = 0.8
p = 200
g = 200

# Taking input from the user
option = int(input("What would you like to find the solution for? "
                   "Type 1 for Eggholder, type 2 for Holder table."))

# Setting the x and y limits
if option == 1:
    lower = -512
    upper = 512
else:
    lower = -10
    upper = 10


# To result the value of the objective function, which gives the fitness of a point
def objective_function(xig):

    x = xig[0]
    y = xig[1]

    # Egg holder function
    if option == 1:
        return -(y+47)*math.sin(math.sqrt(abs(x/2 + (y+47)))) - x*math.sin(math.sqrt(abs(x-(y+47))))

    # Holder table function
    if option == 2:
        return -abs(math.sin(x)*math.cos(y)*math.exp(abs(1 - (math.sqrt(x**2 + y**2))/math.pi)))


# To evaluate the mutant vector
def mutant_vector(xig, population, f):

    x = xig[0]
    y = xig[1]

    # Choosing three random candidates from the population
    index1, index2, index3 = random.choices(range(len(population)), k=3)
    x1, y1 = population[index1][0], population[index1][1]
    x2, y2 = population[index2][0], population[index2][1]
    x3, y3 = population[index3][0], population[index3][1]

    return [x + K*(x1 - x) + f*(x2 - x3), y + K*(y1 - y) + f*(y2 - y3)]


# To form the trial vector
def trial_vector(xig, vig):

    zig = []

    for j in range(len(xig)):
        if random.random() <= Cr:
            zig.append(vig[j])
        else:
            zig.append(xig[j])

    return zig


def constraints_satisfied(xig):

    # Checking if x and y are in the given range
    if xig[0] > upper or xig[0] < lower or xig[1] > upper or xig[1] < lower:
        return False
    else:
        return True


# To check if the trial vector has better fitness than the original vector or not
def is_fitter(new_xig, old_xig):
    if objective_function(old_xig) < objective_function(new_xig):
        return False
    else:
        return True


# To find best candidates, best fitness and average fitness of the population in a generation
def best_of_all(population):

    fitness_list = []

    for member in range(len(population)):
        fitness_list.append(objective_function(population[member]))

    best_positions = []

    # To find candidates having the best fitness
    for member in range(len(population)):
        if fitness_list[member] == min(fitness_list):
            best_positions.append(population[member])

    return best_positions, min(fitness_list), sum(fitness_list)/len(population)


# To define a population
def generate_population():

    # Choosing p random x and y
    x = np.random.uniform(lower, upper, size=p)
    y = np.random.uniform(lower, upper, size=p)

    population = []

    # Forming p vectors using the above random x and y
    for member in range(p):
        population.append([x[member], y[member]])

    return population


# To plot a set of vectors
def plot(population, symbol, label):

    x = []
    y = []

    # Taking the elements of vectors into x and y lists
    for member in range(len(population)):
        x.append(population[member][0])
        y.append(population[member][1])

    plt.plot(x, y, symbol, label=label)


# MAIN ALGORITHM:

# Generating a population
candidates = generate_population()

# Plotting the initial population
plt.title("Initial Population")
plot(candidates, "+", "Candidate")
plt.show()

# Creating lists for best candidates, best fitness and average fitness in a population
best_candidates = [best_of_all(candidates)[0]]
best_fitness = [best_of_all(candidates)[1]]
average_fitness = [best_of_all(candidates)[2]]

# Creating a for loop to run across G given generations
for gen in range(g):

    # Initializing the candidate index
    i = 0

    # Generating a random F in the range -2 to 2, for every generation
    F = random.uniform(-2, 2)

    # Creating a while loop to run across all the candidates in the population
    while i in range(p):

        # Getting the vector of the ith candidate in the population
        x_ig = candidates[i]

        # Generating the mutant vector for the candidate xig
        v_ig = mutant_vector(x_ig, candidates, F)

        # Generating the trial vector
        z_ig = trial_vector(x_ig, v_ig)

        # Checking if the constraints are satisfied
        if not constraints_satisfied(z_ig):
            # If not satisfied, form new mutant and trail vectors again
            continue

        # Checking if the trial vector has better fitness than the original
        if not is_fitter(z_ig, x_ig):
            # If it is not, move on to the next candidate
            i = i+1
            continue

        # Replacing the candidate with its trial vector because of the better fitness
        candidates[i] = z_ig

        # Incrementing the candidate index to go to the next candidate
        i = i+1

    # Finding best candidates, best fitness and average fitness in every generation
    best_candidates, f_best, f_avg = best_of_all(candidates)

    # Appending them to the created lists
    best_fitness.append(f_best)
    average_fitness.append(f_avg)


# PLOTTING:

# Plotting the population after the evolution, along with the best candidate
plt.title("Solution")
plot(candidates, "+", "Candidate")
plot(best_candidates, "*r", "Best candidate")
plt.legend(loc="upper left")
plt.xlim(lower, upper)
plt.ylim(lower, upper)
plt.show()

# Plotting the best and average fitness across generations
plt.title("Convergence history")
plt.plot(best_fitness, label="Best fitness")
plt.plot(average_fitness, label="Average fitness")
plt.legend(loc="upper left")
plt.show()

# Outputting the best fitness/ minimum of the function and the best candidates.
print(f"Minimum = {min(best_fitness)}\nPoint(s):{best_candidates}")
