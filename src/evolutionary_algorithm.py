import random

from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split

from src.data_loader import SMSDataLoader
from src.feature_extractor import FeatureExtractor
from src.fuzzy_sets import *

RULE_SIZE = 11
NUM_RULES = 50
LINGUISTIC_VALUES = {
    0: lambda x: iso_tri(x, 0, 1),
    1: lambda x: rect_trap(x, 1, 1),
    2: lambda x: gaussian(x, 0, 1),
    3: lambda x: sigmoid(x, 0, 1),
}
POPULATION_SIZE = 10
NUM_GENERATIONS = 10
CROSSOVER_PROBABILITY = 0.6
MUTATION_PROBABILITY = 0.1


class FuzzyRuleBase:
    def __init__(self, rule_base, linguistic_values):
        self.rules = rule_base
        self.linguistic_values = linguistic_values

    def calc_rule_compatibility(self, sample, rule):
        compat = 1
        for i in range(0, len(rule) - 1, 2):
            lv = rule[i]
            neg = rule[i + 1]
            if lv in self.linguistic_values:
                value = self.linguistic_values[lv](sample[i // 2])
                if neg:
                    value = negation(value)
                compat *= value
        return compat

    def calc_gc(self, sample, rules, c):
        compatibilities = [self.calc_rule_compatibility(sample, rule) for rule in rules]
        return sum([compatibilities[j] for j in range(len(compatibilities)) if rules[j][-1] == c])

    def calc_fc(self, x, y, rule, c):
        compatibilities = [self.calc_rule_compatibility(x[i], rule) for i in range(x.shape[0])]
        return sum([compatibilities[i] for i in range(len(compatibilities)) if y[i] == c])

    def calc_eternity_factor(self, x, y, rule):
        num_classes = len(np.unique(y))
        fc_values = [self.calc_fc(x, y, rule, c) for c in range(num_classes)]
        fyj = fc_values[rule[-1]]
        fneg = (1 / (num_classes - 1)) * sum([fc_values[i] for i in range(num_classes) if i != rule[-1]])
        return (fyj - fneg) / (sum(fc_values) + 0.01)

    def predict(self, sample, rules):
        num_classes = len(np.unique([rule[-1] for rule in rules]))
        gc_values = [self.calc_gc(sample, rules, c) for c in range(num_classes)]
        return np.argmax(gc_values)

    def evaluate(self, x, y, rule_base):
        predictions = [self.predict(x[i], rule_base) for i in range(x.shape[0])]
        accuracy = sum([1 if predictions[i] == y[i] else 0 for i in range(len(y))]) / len(y)
        return accuracy


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
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("attr_linguistic", random.randint, 0, 3)
        toolbox.register("attr_negation", random.randint, 0, 1)
        toolbox.register("attr_class", random.randint, 0, 1)
        toolbox.register("individual", tools.initCycle, creator.Individual,
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
        fuzzy_rule_system = FuzzyRuleBase(rule_base, LINGUISTIC_VALUES)
        cf_values = [fuzzy_rule_system.calc_eternity_factor(self.x_train, self.y_train, rule) for rule in rule_base]
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


def main():
    data_loader = SMSDataLoader()
    records, labels = data_loader.load_data('SMSSpamCollection')
    feature_extractor = FeatureExtractor()
    features = feature_extractor.select_features(records, labels)
    labels = np.array(labels, dtype=int)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    evolutionary_algorithm = EvolutionaryAlgorithm(RULE_SIZE, NUM_RULES, x_train, y_train, LINGUISTIC_VALUES)
    best_rule_base = evolutionary_algorithm.run(
        POPULATION_SIZE, NUM_GENERATIONS, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY
    )

    print("Best rule base:", best_rule_base)

    fuzzy_rule_system = FuzzyRuleBase(best_rule_base, LINGUISTIC_VALUES)

    accuracy = fuzzy_rule_system.evaluate(x_train, y_train, best_rule_base)
    print(f"Accuracy on the train set: {accuracy * 100:.2f}%")

    accuracy = fuzzy_rule_system.evaluate(x_test, y_test, best_rule_base)
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
