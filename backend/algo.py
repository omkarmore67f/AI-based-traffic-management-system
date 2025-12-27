import numpy as np

# ---------------- FITNESS FUNCTION ----------------
def fitness_function(C, g, x, c):
    if C <= 0 or c <= 0:
        return 0
    a = (1 - (g / C)) ** 2
    p = max(0.1, 1 - ((g / C) * x))
    d1i = (0.38 * C * a) / p
    d2i = 173 * (x ** 2)
    return d1i + d2i


# ---------------- GENETIC ALGORITHM ----------------
def genetic_algorithm(pop_size, num_lights, max_iter, green_min, green_max,
                      cycle_time, mutation_rate, pinv, beta, cars):

    cars = np.array(cars, dtype=float)
    cars[cars <= 0] = 1.0  # safety

    road_capacity = np.array([20] * num_lights)
    congestion = (road_capacity - cars) / road_capacity
    congestion = np.clip(congestion, 0.05, 1)

    # Initial population
    population = []
    while len(population) < pop_size:
        green = np.random.randint(green_min, green_max, num_lights)
        if green.sum() <= cycle_time:
            delay = sum(
                fitness_function(cycle_time, green[i], congestion[i], road_capacity[i])
                for i in range(num_lights)
            )
            population.append((green, delay))

    population.sort(key=lambda x: x[1])
    best = population[0]

    # Evolution
    for _ in range(max_iter):
        new_pop = []
        for _ in range(pop_size):
            p1, p2 = population[np.random.randint(pop_size)][0], population[np.random.randint(pop_size)][0]
            cut = np.random.randint(1, num_lights)
            child = np.concatenate([p1[:cut], p2[cut:]])
            child = np.clip(child, green_min, green_max)

            if child.sum() <= cycle_time:
                delay = sum(
                    fitness_function(cycle_time, child[i], congestion[i], road_capacity[i])
                    for i in range(num_lights)
                )
                new_pop.append((child, delay))

        population = sorted(population + new_pop, key=lambda x: x[1])[:pop_size]
        best = population[0]

    return best


# ---------------- OPTIMIZER ----------------
def optimize_traffic(cars):
    cars = [max(1.0, float(c)) for c in cars]

    pop_size = 120
    num_lights = 4
    max_iter = 15
    green_min = 10
    green_max = 45
    cycle_time = 120
    mutation_rate = 0.02
    pinv = 0.2
    beta = 6

    best_sol = genetic_algorithm(
        pop_size, num_lights, max_iter,
        green_min, green_max, cycle_time,
        mutation_rate, pinv, beta, cars
    )

    return {
        "L1": {"green": int(best_sol[0][0])},
        "L2": {"green": int(best_sol[0][1])},
        "L3": {"green": int(best_sol[0][2])},
        "L4": {"green": int(best_sol[0][3])}
    }
