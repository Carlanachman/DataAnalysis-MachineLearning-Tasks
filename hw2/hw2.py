import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    

    d = data[:, -1]
    n = len(d)

    if n == 0:
      return 0.0
    
    cpt = Counter(d)
    for c in cpt.values():
      p = c/n
      gini += p*p

    gini = 1 - gini

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    

    d = data[:, -1]
    n = len(d)

    if n == 0:
      return 0.0
    
    cpt = Counter(d)
    for c in cpt.values():
      p = c/n
      entropy -= p* np.log2(p)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

class DecisionNode:

    
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        d = self.data[:, -1]
        most_common = Counter(d).most_common(1)
        pred = most_common[0][0] if most_common else None

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        self.children.append(node)
        self.children_values.append(val)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
    
        feature = self.feature
        n = len(self.data)
        goodness, _ = self.goodness_of_split(feature)
        self.feature_importance = (n / n_total_sample) * goodness

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        S = self.data
        impurity_S = self.impurity_func(S)
        values = np.unique(S[:, feature])
        w_impurity = 0

        for v in values:
          subset = S[S[:, feature] == v]
          groups[v] = subset
          w = len(subset) / len(S)
          w_impurity += w * self.impurity_func(subset)

        goodness = impurity_S - w_impurity

        if self.gain_ratio:
            split_info = 0
            for val in groups:
                p = len(groups[val]) / len(S)
                if p > 0: 
                    split_info -= p * np.log2(p)
            goodness = goodness / split_info if split_info != 0 else 0 

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return goodness, groups
    
    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        if self.depth >= self.max_depth or self.impurity_func(self.data) == 0 :
          self.terminal = True
          return

        best_gain = -1
        best_feature = -1
        best_groups = None

        for feature in range(self.data.shape[1] - 1):
          gain, groups = self.goodness_of_split(feature)
          if gain > best_gain:
            best_gain = gain 
            best_feature = feature
            best_groups = groups
        

        if best_gain <= 0 or best_groups is None:
          self.terminal= True 
          return
        
        self.prone(best_groups)
        if self.terminal:
            return

        self.feature = best_feature

        for val, group in best_groups.items():
          child = DecisionNode(
            data = group,
            impurity_func = self.impurity_func,
            feature = -1,
            depth = self.depth + 1,
            chi = self.chi,
            max_depth = self.max_depth, 
            gain_ratio = self.gain_ratio
          )

          self.add_child(child, val)
          child.split()


        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def prone(self, groups):

        n_groups = len(groups)
        chi_statistic = 0

        label_counts = Counter(self.data[:, -1])
        total_labels = sum(label_counts.values())
        label_probabilities = {label: count / total_labels for label, count in label_counts.items()}

        for group in groups.values():
            n_group = len(group)
            if n_group > 0:
                group_label_counts = Counter(group[:, -1])

                for label, p_y in label_probabilities.items():
                    observed = group_label_counts.get(label, 0)
                    expected = n_group * p_y

                    if expected > 0:
                        chi_statistic += ((observed - expected) ** 2) / expected

        degrees_of_freedom = (n_groups - 1) * (len(label_counts) - 1)
        critical_value = chi_table.get(degrees_of_freedom, {}).get(self.chi, float('-inf'))

        if chi_statistic < critical_value:
            self.terminal = True
            return
        ###########################################################################


class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree
        
    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        self.root = DecisionNode(
            data = self.data,
            impurity_func = self.impurity_func,
            feature = -1,
            depth = 0,
            chi = self.chi,
            max_depth = self.max_depth, 
            gain_ratio = self.gain_ratio
        )
        self.root.split()


        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.root is None:
            raise ValueError("Tree has not been built yet. Please call build_tree() first.")
        node = self.root
        while not node.terminal:
            feature = node.feature
            value = instance[feature]
            found = False
            for i, child in enumerate(node.children):
                if node.children_values[i] == value:
                    node = child
                    found = True
                    break
            if not found:
                break
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.root is None:
            raise ValueError("Tree has not been built yet. Please call build_tree() first.")
        n = len(dataset)
        correct = 0
        for i in range(n):
            pred = self.predict(dataset[i])
            if pred == dataset[i][-1]:
                correct += 1
        accuracy = (correct / n)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy
        
    def depth(self):
        return self.root.depth()

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        tree = DecisionTree(
            data = X_train,
            impurity_func = calc_entropy,
            feature = -1,
            chi = 1,
            max_depth = max_depth, 
            gain_ratio = True
        )
        tree.build_tree()
        training_acc = tree.calc_accuracy(X_train)
        validation_acc = tree.calc_accuracy(X_validation)
        training.append(training_acc)
        validation.append(validation_acc)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for chi in [1 ,0.5, 0.25, 0.1, 0.05, 0.0001]:
        tree = DecisionTree(
            data = X_train,
            impurity_func = calc_entropy,
            feature = -1,
            chi = chi,
            max_depth = 1000, 
            gain_ratio = True
        )
        tree.build_tree()
        training_acc = tree.calc_accuracy(X_train)
        testing_acc = tree.calc_accuracy(X_test)
        depth.append(count_nodes(tree.root))
        chi_training_acc.append(training_acc)
        chi_validation_acc.append(testing_acc)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
        
    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    n_nodes = 1
    if node.terminal:
        return n_nodes
    for child in node.children:
        n_nodes += count_nodes(child)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






