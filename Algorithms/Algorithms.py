import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Количество городов
NUM_CITIES = 15

# Создание случайных координат городов
CITIES = {i: (random.randint(0, 100), random.randint(0, 100)) for i in range(NUM_CITIES)}

# Вычисление расстояния между двумя городами
def distance(city1, city2):
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# Создание класса FitnessMin для минимизации длины маршрута
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Генерация случайного индивида (маршрута)
toolbox.register("indices", random.sample, range(NUM_CITIES), NUM_CITIES)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Вычисление длины маршрута
def total_distance(individual):
    return sum(distance(CITIES[individual[i]], CITIES[individual[i + 1]]) for i in range(len(individual) - 1)) + distance(CITIES[individual[-1]], CITIES[individual[0]])

# Регистрация операторов для генетического алгоритма
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", total_distance)

def main():
    random.seed(42)

    # Генерация начальной популяции
    pop = toolbox.population(n=300)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 100

    # Запуск генетического алгоритма
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (fit,)

    for g in range(NGEN):
        # Выбор следующего поколения особей
        offspring = toolbox.select(pop, len(pop))
        # Клонирование выбранных особей
        offspring = list(map(toolbox.clone, offspring))

        # Применение скрещивания и мутации для создания новых особей
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Оценка особей с недопустимыми значениями фитнеса
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        # Объединение популяций
        pop[:] = offspring

    # Получение лучшей особи и вывод кратчайшего пути
    best_ind = tools.selBest(pop, 1)[0]
    best_route = [list(CITIES.keys())[list(CITIES.values()).index(CITIES[best_ind[i]])] for i in range(NUM_CITIES)]
    print("short:", best_route)
    print("long:", total_distance(best_route))

    # Построение графика маршрута
    x = [CITIES[i][0] for i in best_route]
    y = [CITIES[i][1] for i in best_route]
    plt.plot(x, y, 'ro-')
    for i, txt in enumerate(best_route):
        plt.annotate(txt, (x[i], y[i]), xytext=(5, -5), textcoords='offset points')
    plt.title("short")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

if __name__ == "__main__":
    main()

