from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

@dataclass
class Node:
    attribute: Optional[str] = None
    value: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    is_leaf: bool = False
    output: Optional[Union[float, str]] = None

    def is_leaf_node(self) -> bool:
        return self.is_leaf

class DecisionTree:
    criterion: Literal["variance", "gini", "mse"]
    max_depth: int

    def __init__(self, criterion: Literal["variance", "gini", "mse"], max_depth: int = 5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> Node:
        """
        Function to train and construct the decision tree.
        """

        # If the depth exceeds max_depth or all the target values are the same, create a leaf node
        if depth >= self.max_depth or y.nunique() == 1:
            if check_ifreal(y):
                return Node(is_leaf=True, output=y.mean())
            else:
                return Node(is_leaf=True, output=y.mode()[0])

        best_attribute = opt_split_attribute(X, y, X.columns, self.criterion)

        # If no good split is found, create a leaf node
        if best_attribute is None:
            if check_ifreal(y):
                return Node(is_leaf=True, output=y.mean())
            else:
                return Node(is_leaf=True, output=y.mode()[0])

        if check_ifreal(X[best_attribute]):
            best_value = real_feature_thresholding(y, X[best_attribute], self.criterion)
        else:
            best_value = X[best_attribute].mode()[0]

        X_left, y_left, X_right, y_right = split_data(X, y, best_attribute, best_value)

        # If a valid split is not possible, return a leaf node
        if X_left.empty or X_right.empty:
            if check_ifreal(y):
                return Node(is_leaf=True, output=y.mean())
            else:
                return Node(is_leaf=True, output=y.mode()[0])

        left_subtree = self.fit(X_left, y_left, depth + 1)
        right_subtree = self.fit(X_right, y_right, depth + 1)

        return Node(attribute=best_attribute, value=best_value, left=left_subtree, right=right_subtree)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Function to run the decision tree on test inputs.
        """

        def predict_single(node: Node, x: pd.Series) -> Union[float, str]:
            if node.is_leaf_node():
                return node.output
            if check_ifreal(X[node.attribute]):
                if x[node.attribute] <= node.value:
                    return predict_single(node.left, x)
                else:
                    return predict_single(node.right, x)
            else:
                if x[node.attribute] == node.value:
                    return predict_single(node.left, x)
                else:
                    return predict_single(node.right, x)

        return X.apply(lambda x: predict_single(self.tree, x), axis=1)

    def plot(self) -> None:
        """
        Function to plot the tree.
        """
        def plot_tree(node: Node, depth: int = 0):
            if node.is_leaf_node():
                print(f"{' ' * depth * 4}Leaf: {node.output}")
            else:
                print(f"{' ' * depth * 4}{node.attribute} <= {node.value}")
                print(f"{' ' * (depth + 1) * 4}Left:")
                plot_tree(node.left, depth + 1)
                print(f"{' ' * (depth + 1) * 4}Right:")
                plot_tree(node.right, depth + 1)

        if self.tree is not None:
            plot_tree(self.tree)
        else:
            print("Tree has not been fitted yet.")

