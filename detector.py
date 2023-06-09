import random

import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

rule_size = 11
num_rules = 50


# Fuzzy sets
def iso_tri(x, m, s):
    return max(min((x - m + s) / s, (m - x + s) / s), 0)


def rect_trap(x, m, s):
    return max(min((x - m + s) / s, 1), 0)


def gaussian(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2)


def sigmoid(x, m, s):
    return 1 / (1 + np.exp(-(x - m) / s))


def negation(x):
    return 1 - x


# Calculate rule compatibility
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


# Compute gc(x(p))
def gc(sample, rules, linguistic_values, c):
    compatibilities = [calculate_gR(sample, rule, linguistic_values) for rule in rules]
    return sum([compatibilities[j] for j in range(len(compatibilities)) if rules[j][-1] == c])


# Compute the predicted class ŷ(x(p))
def predict(sample, rules, linguistic_values):
    num_classes = len(np.unique([rule[-1] for rule in rules]))
    gc_values = [gc(sample, rules, linguistic_values, c) for c in range(num_classes)]
    return np.argmax(gc_values)


# Fitness function
def objective_function(individuals, X, y, linguistic_values, rule_size):
    rule_base = [individuals[i:i + rule_size] for i in range(0, len(individuals), rule_size)]
    cf_values = [calculate_cf(X, y, rule, linguistic_values) for rule in rule_base]
    return sum(cf_values) / len(rule_base),


# Calculate the CF values
def calculate_cf(X, y, rule, linguistic_values):
    num_classes = len(np.unique(y))
    fc_values = [fc(X, y, rule, linguistic_values, c) for c in range(num_classes)]
    fyj = fc_values[rule[-1]]
    fneg = (1 / (num_classes - 1)) * sum([fc_values[i] for i in range(num_classes) if i != rule[-1]])
    return (fyj - fneg) / (sum(fc_values) - 1)


# Calculate fc(Rj)
def fc(X, y, rule, linguistic_values, c):
    compatibilities = [calculate_gR(X[i], rule, linguistic_values) for i in range(X.shape[0])]
    return sum([compatibilities[i] for i in range(len(compatibilities)) if y[i] == c])


# Load the dataset, preprocess the data, and split the dataset into training and testing parts

# Load the dataset
sms_data_str = None
with open('SMSSpamCollection') as file:
    sms_data_str = file.read()


# Preprocess the data
def process_data(sms_data_str):
    data_arr = []

    data_records = sms_data_str.split('\n')[:-1]
    for data in data_records:
        label = None
        sample = None
        if data[:3] == 'ham':
            label = 0
            sample = data[4:]
        elif data[:4] == 'spam':
            label = 1
            sample = data[5:]
        else:
            label = 'N/A'

        data_arr.append([label, sample])

    data_arr = np.array(data_arr)
    data_label = data_arr[:, 0]
    data_records = data_arr[:, 1]

    return data_records, data_label


records, labels = process_data(sms_data_str)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r'\b[A-Za-z]+\b', norm=None)
records_transformed = vectorizer.fit_transform(records).toarray()

# One hot encoding labels
labels = np.array(labels, dtype=int)

# Feature extraction
reduction_pca = PCA(n_components=5, whiten=False)
records_dim_reduced = reduction_pca.fit_transform(records_transformed)

# Feature selection
# feature_selection_model = SelectKBest(mutual_info_classif, k=5)
# records_selection = feature_selection_model.fit_transform(records_transformed, labels)

records_selection = records_dim_reduced
# save features and labels to file
np.savetxt("records_selection.csv", records_selection, delimiter=",")
np.savetxt("labels.csv", np.array(labels, dtype=int), delimiter=',')
records_selection = np.loadtxt("records_selection.csv", delimiter=",")
labels = np.loadtxt("labels.csv", delimiter=",").astype(np.int64)

# Load the features and labels and split the dataset into training and testing parts
records_selection = np.loadtxt("records_selection.csv", delimiter=",")
labels = np.loadtxt("labels.csv", delimiter=",").astype(np.int64)

# Split the dataset into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(records_selection, labels, test_size=0.2, random_state=42)

# Linguistic values
linguistic_values = {
    0: lambda x: iso_tri(x, 0, 1),
    1: lambda x: rect_trap(x, 1, 1),
    2: lambda x: gaussian(x, 0, 1),
    3: lambda x: sigmoid(x, 0, 1),
}

# Create the types, the functions, and the genetic algorithm parameters

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
toolbox.register("evaluate", objective_function, X=X_train, y=y_train, linguistic_values=linguistic_values,
                 rule_size=rule_size)

# Genetic algorithm parameters
population_size = 10
num_generations = 10
crossover_probability = 0.7
mutation_probability = 0.3

# Create the initial population
population = toolbox.population(n=population_size)

# Perform the genetic algorithm
hof = tools.HallOfFame(10)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

population, logbook = algorithms.eaSimple(population, toolbox, cxpb=crossover_probability, mutpb=mutation_probability,
                                          ngen=num_generations, stats=stats, halloffame=hof, verbose=True)

# Test the best individual on the test set
best_individual = hof[0]
best_rule_base = [best_individual[i:i + rule_size] for i in range(0, len(best_individual), rule_size)]
print("Best rule base:", best_rule_base)

predictions = [predict(X_test[i], best_rule_base, linguistic_values) for i in range(X_test.shape[0])]
accuracy = sum([1 if predictions[i] == y_test[i] else 0 for i in range(len(y_test))]) / len(y_test)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
