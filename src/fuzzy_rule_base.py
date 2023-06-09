import numpy as np

from src.fuzzy_sets import negation


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
