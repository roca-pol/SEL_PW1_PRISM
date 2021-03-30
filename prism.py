import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


class Prism:

    _op_map = {
        '>=': lambda a, b: a >= b,
        '<': lambda a, b: a < b,
        '==': lambda a, b: a == b
    }

    def __init__(self, filename=None):
        if filename is not None:
            self._parse_file(filename)

    def fit(self, data: pd.DataFrame, target='class', n_bins=3):

        # discretize numerical attributes if there are any
        num_attr = data.select_dtypes(include=['number']).columns
        num_attr = num_attr.drop(target) if target in num_attr else num_attr
        if len(num_attr) > 0:
            data = data.copy()
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal') # , strategy='uniform')
            data[num_attr] = discretizer.fit_transform(data[num_attr]).astype(np.int)
            bin_edges = discretizer.bin_edges_

        data = data.drop_duplicates(data.columns.drop(target))

        # count how many instances each class has
        classes, counts = np.unique(data[target], return_counts=True)
        ruleset = []
        all_attr = data.columns.drop(target)

        # main loop - generate rules for each class
        for class_, unclass_count in zip(classes, counts):
            instance_set = data
            total_tp = unclass_count

            while unclass_count > 0:
                rule = []
                unused_attr = list(all_attr)
                rule_coverage = instance_set
                precision = 0

                # construct the rule by adding selectors to the antecedent
                while len(unused_attr) > 0 and precision != 1:
                    precision, best_tp = 0, 0
                    best_attr, best_value = None, None
                    best_selector = None

                    # look for the best attribute-value pair in terms of precision
                    for attr in unused_attr:
                        for value in rule_coverage[attr].unique():
                            selector = rule_coverage[attr].values == value
                            tp = (rule_coverage[target][selector].values == class_).sum()
                            tp_fp = selector.sum()
                            selector_precision = tp / tp_fp
                            if selector_precision > precision or selector_precision == precision and tp > best_tp:
                                precision = selector_precision
                                best_attr, best_value = attr, value
                                best_tp = tp
                                best_selector = selector

                    rule_coverage = rule_coverage[best_selector]
                    unused_attr.remove(best_attr)
                    
                    # append the best selector to the antecedent of the rule
                    if best_attr in num_attr:
                        idx = num_attr.get_loc(best_attr)
                        edges = bin_edges[idx]
                        if best_value == 0:  # lower interval
                            rule.append((best_attr, '<', edges[1]))
                        elif best_value == len(edges)-2:  # higher interval
                            rule.append((best_attr, '>=', edges[-2]))
                        else:  # anything inbetween
                            rule.append((best_attr, '>=', edges[best_value]))
                            rule.append((best_attr, '<', edges[best_value+1]))
                    else:
                        rule.append((best_attr, '==', best_value))

                rule.append((class_, precision, best_tp / total_tp))
                ruleset.append(rule)
                instance_set = instance_set.drop(rule_coverage.index)
                unclass_count -= best_tp

        self.ruleset_ = ruleset
        self.target_ = target
        self.majority_ = data[target].mode().values[0]

    def predict(self, data: pd.DataFrame):        
        labels = []
        for _, ins in data.iterrows():
            label = None
            for rule in self.ruleset_:
                if self._apply_rule(rule, ins):
                    label = rule[-1][0]
                    break
            labels.append(label or self.majority_)
                
        return np.array(labels)

    def _apply_rule(self, rule, instance):
        for attr, op_str, value in rule[:-1]:
            op = self._op_map[op_str]
            if not op(instance[attr], value):
                return False
        return True

    def _parse_file(self, filename):
        with open(filename, 'r') as f:
            lines = f.read().strip().splitlines()

        ruleset = []
        target = None
        
        try:
            for line in lines:
                if_part, then_part = line.split(') THEN (')
                antecedent = if_part.split(': IF (')[1]
                selectors = antecedent.split(') AND (')

                # create rule by parsing selectors
                rule = []
                for sel in selectors:
                    for op in self._op_map.keys():
                        if op in sel:
                            attr, val = sel.split(f' {op} ')
                            rule.append((attr, op, self._parse_value(val)))
                            break
                
                if target is None:
                    target = then_part.split(' == ')[0]

                class_ = self._parse_value(then_part.split(' == ')[1].split(') [')[0])
                precision = float(then_part.split(') [')[1].split(']')[0])
                rule.append((class_, precision))
                ruleset.append(rule)
        except:
            raise Exception('PRISM: Error while parsing rules file.')

        self.ruleset_ = ruleset
        self.target_ = target

    def _parse_value(self, value_str):
        try:
            return int(value_str)
        except (ValueError, TypeError):
            try:
                return float(value_str)
            except (ValueError, TypeError):
                return value_str

    def __str__(self):
        if hasattr(self, 'ruleset_'):
            text = ''
            for i, rule in enumerate(self.ruleset_):
                selectors = []
                for attr, op, val in rule[:-1]:
                    if op == '==':
                        selectors.append((attr, op, val))
                    else:
                        selectors.append((attr, op, round(val, 2)))

                antecedent = ' AND '.join([f'({attr} {op} {val})' for attr, op, val in selectors])
                consequent = str(rule[-1][0])
                precision = round(rule[-1][1], 2)
                recall = round(rule[-1][2], 2)
                
                text += padding(i+1, len(self.ruleset_))
                text += f'{i+1}: IF {antecedent} THEN ({self.target_} == {consequent}) [{precision}, {recall}]\n'

            return text
        else:
            return object.__str__(self)


def padding(i, max_num):
    pad = ''
    for exp in range(1, len(str(int(max_num)))):
        if i < 10 ** exp:
            pad += ' '
    return pad