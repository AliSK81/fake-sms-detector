import numpy as np
from sklearn.model_selection import train_test_split

from src.data_loader import SMSDataLoader
from src.evolutionary_algorithm import EvolutionaryAlgorithm
from src.feature_extractor import FeatureExtractor
from src.fuzzy_rule_base import FuzzyRuleBase
from src.fuzzy_sets import iso_tri, rect_trap, gaussian, sigmoid

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
MUTATION_PROBABILITY = 0.2


def main():
    data_loader = SMSDataLoader()
    records, labels = data_loader.load_data('SMSSpamCollection')
    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract_features(records, n_components=(NUM_RULES - 1) // 2)
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
