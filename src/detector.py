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
CROSSOVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.3


def calculate_gR(sample, rule, linguistic_values):
    compat = 1
    for i in range(0, len(rule) - 1, 2):
        lv = rule[i]
        neg = rule[i + 1]
        if lv in linguistic_values:
            value = linguistic_values[lv](sample[i // 2])
            if neg:
                value = negation(value)
            compat *= value
    return compat


def gc(sample, rules, linguistic_values, c):
    compatibilities = [calculate_gR(sample, rule, linguistic_values) for rule in rules]
    return sum([compatibilities[j] for j in range(len(compatibilities)) if rules[j][-1] == c])


def predict(sample, rules, linguistic_values):
    num_classes = len(np.unique([rule[-1] for rule in rules]))
    gc_values = [gc(sample, rules, linguistic_values, c) for c in range(num_classes)]
    return np.argmax(gc_values)


def calculate_cf(X, y, rule, linguistic_values):
    num_classes = len(np.unique(y))
    fc_values = [fc(X, y, rule, linguistic_values, c) for c in range(num_classes)]
    fyj = fc_values[rule[-1]]
    fneg = (1 / (num_classes - 1)) * sum([fc_values[i] for i in range(num_classes) if i != rule[-1]])
    return (fyj - fneg) / (sum(fc_values) + 0.01)


def fc(X, y, rule, linguistic_values, c):
    compatibilities = [calculate_gR(X[i], rule, linguistic_values) for i in range(X.shape[0])]
    return sum([compatibilities[i] for i in range(len(compatibilities)) if y[i] == c])


def fitness_function(individuals, X, y, linguistic_values, rule_size):
    rule_base = [individuals[i:i + rule_size] for i in range(0, len(individuals), rule_size)]
    cf_values = [calculate_cf(X, y, rule, linguistic_values) for rule in rule_base]
    return sum(cf_values) / len(rule_base),


def create_toolbox(rule_size, num_rules, X_train, y_train, linguistic_values):
    # Create the types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Create the functions
    toolbox = base.Toolbox()
    toolbox.register("attr_linguistic", random.randint, 0, 3)
    toolbox.register("attr_negation", random.randint, 0, 1)
    toolbox.register("attr_class", random.randint, 0, 1)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_linguistic, toolbox.attr_negation) * ((rule_size - 1) // 2) + (toolbox.attr_class,),
                     n=num_rules)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", fitness_function, X=X_train, y=y_train, linguistic_values=linguistic_values,
                     rule_size=rule_size)
    return toolbox


def main():
    data_loader = SMSDataLoader()
    records, labels = data_loader.load_data('SMSSpamCollection')
    feature_extractor = FeatureExtractor()
    features = feature_extractor.select_features(records, labels)

    labels = np.array(labels, dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    toolbox = create_toolbox(RULE_SIZE, NUM_RULES, X_train, y_train, LINGUISTIC_VALUES)

    population = toolbox.population(n=POPULATION_SIZE)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=CROSSOVER_PROBABILITY,
                                              mutpb=MUTATION_PROBABILITY,
                                              ngen=NUM_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    best_individual = hof[0]
    best_rule_base = [best_individual[i:i + RULE_SIZE] for i in range(0, len(best_individual), RULE_SIZE)]
    print("Best rule base:", best_rule_base)

    predictions = [predict(X_test[i], best_rule_base, LINGUISTIC_VALUES) for i in range(X_test.shape[0])]
    accuracy = sum([1 if predictions[i] == y_test[i] else 0 for i in range(len(y_test))]) / len(y_test)
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
