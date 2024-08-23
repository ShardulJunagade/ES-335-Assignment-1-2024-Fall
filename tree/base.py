"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


@dataclass
class Node:
    attribute: str
    value: float
    left: "Node"
    right: "Node"
    is_leaf: bool
    output: float

    def __init__(self, attribute=None, value=None, left=None, right=None, is_leaf=False, output=None):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.output = output
    
    def is_leaf_node(self):
        return self.is_leaf



class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree.


        # If the depth exceeds max_depth or all the target values are the same, create a leaf node
        def build_tree(X: pd.DataFrame, y: pd.Series, depth: int) -> Node:            
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
                best_value = find_optimal_threshold(y, X[best_attribute], self.criterion)
            else:
                best_value = X[best_attribute].mode()[0]

            X_left, y_left, X_right, y_right = split_data(X, y, best_attribute, best_value)

            # If a valid split is not possible, return a leaf node
            if X_left.empty or X_right.empty:
                if check_ifreal(y):
                    return Node(is_leaf=True, output=y.mean())
                else:
                    return Node(is_leaf=True, output=y.mode()[0])
                
            left_subtree = build_tree(X_left, y_left, depth + 1)
            right_subtree = build_tree(X_right, y_right, depth + 1)

            return Node(attribute=best_attribute, value=best_value, left=left_subtree, right=right_subtree)

        self.tree = build_tree(X, y, depth)
    
    def predict_row(self, x: pd.Series) -> float:
        """
        Function to predict the output for a single row of input
        """

        # Traverse the tree you constructed to return the predicted value for the given input.

        current_node = self.tree

        while not current_node.is_leaf_node():
            if check_ifreal(x[current_node.attribute]):
                if x[current_node.attribute] <= current_node.value:
                    current_node = current_node.left
                else:
                    current_node = current_node.right
            else:
                if x[current_node.attribute] == current_node.value:
                    current_node = current_node.left
                else:
                    current_node = current_node.right

        return current_node.output
        


    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        # def predict_single(node: Node, x: pd.Series) -> Union[float, str]:
        #     if node.is_leaf_node():
        #         return node.output
        #     if check_ifreal(X[node.attribute]):
        #         if x[node.attribute] <= node.value:
        #             return predict_single(node.left, x)
        #         else:
        #             return predict_single(node.right, x)
        #     else:
        #         if x[node.attribute] == node.value:
        #             return predict_single(node.left, x)
        #         else:
        #             return predict_single(node.right, x)

        # return X.apply(lambda x: predict_single(self.tree, x), axis=1)
    
        # return X.apply(self.predict_row, axis=1)
        return pd.Series([self.predict_row(x) for _, x in X.iterrows()])

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        pass
