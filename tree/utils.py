"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import numpy as np
import pandas as pd

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data

    Returns the one hot encoded data
    """

    X_encoded = pd.get_dummies(X)
    
    return X_encoded


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values

    Returns True if the series has real (continuous) values, False otherwise (discrete).
    """

    # If the data is float, it's real
    if pd.api.types.is_float_dtype(y):
        return True

    # If the data is integer, it can be either real or discrete
    if pd.api.types.is_integer_dtype(y):
        # If the integer values are limited in number and not too many unique values, it's discrete
        return len(y.unique()) > 10

    # If the data is string, it's discrete
    if pd.api.types.is_string_dtype(y):
        return False
    
    # Else, it's discrete
    return False


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy

    entropy = -sum(p_i * log2(p_i))
    """
    
    value_counts = Y.value_counts()
    total_count = Y.size

    prob = value_counts / total_count
    entropy_value = -np.sum(prob * np.log2(prob))

    return entropy_value


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """



def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """




def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).




def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

  
