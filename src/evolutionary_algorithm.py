import random

from deap import base, creator, tools, algorithms
from src.fuzzy_rule_base import FuzzyRuleBase
from src.fuzzy_sets import *


class EvolutionaryAlgorithm:
    def __init__(self, rule_size, num_rules, x_train, y_train, linguistic_values):
        self.rule_size = rule_size
        self.num_rules = num_rules
        self.x_train = x_train
        self.y_train = y_train
        self.linguistic_values = linguistic_values
        self.toolbox = self.create_toolbox()

    def create_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Chromosome", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("attr_linguistic", random.randint, 0, 3)
        toolbox.register("attr_negation", random.randint, 0, 1)
        toolbox.register("attr_class", random.randint, 0, 1)
        toolbox.register("individual", tools.initCycle, creator.Chromosome,
                         (toolbox.attr_linguistic, toolbox.attr_negation) * ((self.rule_size - 1) // 2) + (
                             toolbox.attr_class,), n=self.num_rules)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=1, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.fitness_function)
        return toolbox

    def fitness_function(self, individuals):
        rule_base = [individuals[i:i + self.rule_size] for i in range(0, len(individuals), self.rule_size)]
        fuzzy_rule_system = FuzzyRuleBase(rule_base, self.linguistic_values)
        cf_values = [fuzzy_rule_system.calc_eternity_factor(self.x_train, self.y_train, rule) for rule in rule_base]
        accuracy = fuzzy_rule_system.evaluate(self.x_train, self.y_train, rule_base)
        print(f"Accuracy on the train set: {accuracy * 100:.2f}%")
        return sum(cf_values) / len(rule_base),

    def run(self, population_size, num_generations, crossover_probability, mutation_probability):
        population = self.toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        algorithms.eaSimple(population, self.toolbox, cxpb=crossover_probability,
                            mutpb=mutation_probability, ngen=num_generations,
                            stats=stats, halloffame=hof, verbose=True)

        best_individual = hof[0]
        best_rule_base = [best_individual[i:i + self.rule_size] for i in range(0, len(best_individual), self.rule_size)]
        return best_rule_base
