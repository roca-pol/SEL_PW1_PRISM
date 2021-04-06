import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

# try to load progress bars module to show a simple display
# with the current progress of the algorithm (OPTIONAL)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class Rule:
    """
    Class that represents a rule.

    Parameters
    ----------
    label : any
        Target class label of the rule.

    Attributes
    ----------
    label : any
        Target class label of the rule.

    antecedent : list
        List of selectors that form the antecedent.

    precision : float
        Precision score of the rule.

    recall : float
        Recall or coverage score of the rule.
    """
    def __init__(self, label):
        self.label = label
        self.antecedent = []
        self.precision = 0.0
        self.recall = 0.0


class Prism:
    """
    PRISM rule inducer algorithm.

    `fit` this classifier to produce a set of rules then use it
    to `predict` labels for a set of instances.
    """
    _op_map = {
        '>=': lambda a, b: a >= b,
        '<': lambda a, b: a < b,
        '==': lambda a, b: a == b
    }

    def fit(self, data: pd.DataFrame, target='class', n_bins=3, strategy='quantile'):
        """
        Induce a ruleset from the given set of instances.

        Parameters
        ----------
        data : DataFrame
            Input training dataset used to induce the rules.
        
        target : str, default='class'
            Name of the attribute that represents class labels.

        n_bins : int, default=3
            Number of bins that numeric attributes will be discretized to.
        
        strategy : {'uniform', 'quantile', 'kmeans'}, default='quantile'
            Strategy used to define the widths of the bins.
            See `sklearn.preprocessing.KBinsDiscretizer` for more info.

        Attributes
        ----------
        ruleset_ : list
            The list of rules induced from the dataset.

        target_ : str
            Name of the attribute that represents class labels.

        majority_ : any
            Label with the largest amount of instances.
        """
        # discretize numerical attributes if there are any
        num_attr = data.select_dtypes(include=['number']).columns
        num_attr = num_attr.drop(target) if target in num_attr else num_attr
        if len(num_attr) > 0:
            data = data.copy()
            discretizer = KBinsDiscretizer(n_bins=n_bins, strategy=strategy,
                                           encode='ordinal')
            with warnings.catch_warnings():
                # sometimes bins are so small that they are merged together
                # that is ok so we do not want to worry about that warning
                warnings.filterwarnings('ignore', category=UserWarning) 
                data[num_attr] = discretizer.fit_transform(
                                    data[num_attr]).astype(np.int8)
            bin_edges = discretizer.bin_edges_

        data = data.drop_duplicates(data.columns.drop(target))

        # count how many instances each class has
        classes, counts = np.unique(data[target], return_counts=True)
        ruleset = []
        all_attr = data.columns.drop(target)

        # prepare a progress bar if user has the module
        pbar = tqdm(total=len(data)) if tqdm is not None else None
        
        # main loop - generate rules for each class
        for label, unclass_count in zip(classes, counts):
            instance_set = data
            total_tp = unclass_count

            while unclass_count > 0:
                rule = Rule(label=label)
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
                            tp = (rule_coverage[target].values[selector] == label).sum()
                            tp_fp = selector.sum()
                            selector_precision = tp / tp_fp
                            if selector_precision > precision or \
                               selector_precision == precision and tp > best_tp:
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
                            rule.antecedent.append((best_attr, '<', edges[1]))
                        elif best_value == len(edges)-2:  # higher interval
                            rule.antecedent.append((best_attr, '>=', edges[-2]))
                        else:  # anything inbetween
                            rule.antecedent.append((best_attr, '>=', edges[best_value]))
                            rule.antecedent.append((best_attr, '<', edges[best_value+1]))
                    else:
                        rule.antecedent.append((best_attr, '==', best_value))

                rule.label = label
                rule.precision = precision
                rule.recall = best_tp / total_tp
                ruleset.append(rule)
                instance_set = instance_set.drop(rule_coverage.index)
                unclass_count -= best_tp

                # update progress bar
                if pbar is not None:
                    pbar.update(best_tp)

        self.ruleset_ = ruleset
        self.target_ = target
        self.majority_ = data[target].mode().values[0]
        return self

    def predict(self, data: pd.DataFrame):        
        """
        Classify all the instances in the provided dataset using the
        rules induced with the previous `fit`.

        Parameters
        ----------
        data : DataFrame, shape (n_samples, n_attributes)
            Set of instances to classify.

        Returns
        -------
        labels : array, shape (n_samples,)
            Class labels of the provided instances.
        """
        if not hasattr(self, 'ruleset_'):
            raise AttributeError("This instance is not fitted yet. Call 'fit' "
                "with the appropriate arguments before using this model.")

        labels = []
        for _, ins in data.iterrows():
            label = None
            for rule in self.ruleset_:
                if self._apply_rule(rule, ins):
                    label = rule.label
                    break
            labels.append(label or self.majority_)
                
        return np.array(labels)

    def _apply_rule(self, rule, instance):
        for attr, op_str, value in rule.antecedent:
            op = self._op_map[op_str]
            if not op(instance[attr], value):
                return False
        return True

    def __str__(self):
        if hasattr(self, 'ruleset_'):
            text = '\n'
            for i, rule in enumerate(self.ruleset_):
                selectors = []
                for attr, op, val in rule.antecedent:
                    if op == '==':
                        selectors.append((attr, op, val))
                    else:
                        selectors.append((attr, op, round(val, 2)))

                antecedent = ' AND '.join([f'({attr} {op} {val})' for attr, op, val in selectors])
                label = str(rule.label)
                precision = round(rule.precision, 2)
                recall = round(rule.recall, 4)
                
                text += padding(i+1, len(self.ruleset_))
                text += f'{i+1}: IF {antecedent} THEN ({self.target_} == {label}) [{precision}, {recall}]\n'
            return text
        else:
            return object.__str__(self)


def padding(i, max_num):
    """Line padding for printing"""
    pad = ''
    for exp in range(1, len(str(int(max_num)))):
        if i < 10 ** exp:
            pad += ' '
    return pad