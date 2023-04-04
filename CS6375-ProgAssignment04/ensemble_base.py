# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# Import Sklearn Libraries for comparison


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    unique_values = np.unique(x)

    dic = {k: [] for k in unique_values}

    for idx, val in enumerate(x):
        dic[val].append(idx)

    return dic


def entropy(y, weights=None):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z. 
    Include the weights of the boosted examples if present

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    unique_values = partition(y)
    number_of_samples = len(y)
    hy = 0
    for elem in unique_values.keys():
        if weights is None:
            p_elem = float(len(unique_values[elem]) / number_of_samples)
        else:
            p_elem = 0
            for idx in unique_values[elem]:
                p_elem += weights[idx]

        log_p_elem = np.log2(p_elem)
        hy += -(p_elem * log_p_elem)
    return hy


def mutual_information(x, y, weights=None):
    """
    
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)

    Compute the weighted mutual information for Boosted learners
    """

    # INSERT YOUR CODE HERE
    hy = entropy(y, weights)
    unique_values_of_x = partition(x)
    number_of_samples = len(x)
    hyx = 0
    for elem in unique_values_of_x.keys():
        if weights is None:
            p_x_elem = float(len(unique_values_of_x[elem]) / number_of_samples)
            weights_new = None
        else:
            p_x_elem = 0
            for idx in unique_values_of_x[elem]:
                p_x_elem += weights[idx]
            weights_new = [weights[i] for i in unique_values_of_x[elem]]
        y_new = [y[i] for i in unique_values_of_x[elem]]
        hyx_elem = entropy(y_new, weights_new)
        hyx += (p_x_elem * hyx_elem)
    return hy - hyx


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5, weights=None):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    decisionTree = {}

    if attribute_value_pairs is None:
        attribute_value_pairs = []
        for i in range(len(x[0])):
            for val in np.unique(np.array([item[i] for item in x])):
                attribute_value_pairs.append((i, val))

    attribute_value_pairs = np.array(attribute_value_pairs)

    # check for pure splits
    unique_values_of_y, count_y = np.unique(y, return_counts=True)
    if len(unique_values_of_y) == 1:
        return unique_values_of_y[0]

    if len(attribute_value_pairs) == 0 or depth == max_depth:
        return unique_values_of_y[np.argmax(count_y)]

    information_gain = []

    for feat, val in attribute_value_pairs:
        information_gain.append(mutual_information(np.array((x[:, feat] == val).astype(int)), y, weights))

    information_gain = np.array(information_gain)
    (feat, val) = attribute_value_pairs[np.argmax(information_gain)]

    partitions = partition(np.array((x[:, feat] == val).astype(int)))

    attribute_value_pairs = np.delete(attribute_value_pairs, np.argmax(information_gain), 0)

    for value, indices in partitions.items():
        x_new = x.take(np.array(indices), axis=0)
        y_new = y.take(np.array(indices), axis=0)
        output = bool(value)
        if weights is None:
            decisionTree[(feat, val, output)] = id3(x_new, y_new, attribute_value_pairs=attribute_value_pairs,
                                                    depth=depth + 1, max_depth=max_depth)
        else:
            weights_new = weights.take(np.array(indices), axis=0)
            decisionTree[(feat, val, output)] = id3(x_new, y_new, attribute_value_pairs=attribute_value_pairs,
                                                    depth=depth + 1, max_depth=max_depth, weights=weights_new)

    return decisionTree


def bootstrap_sampler(x, y, num_samples):
    indices = np.random.choice(num_samples, num_samples, replace=True)
    L = x[indices].astype(int)
    y_l = y[indices].astype(int)
    return L, y_l


def bagging(x, y, max_depth, num_trees):
    """
    Implements bagging of multiple id3 trees where each tree trains on a boostrap sample of the original dataset
    """

    h_ens = []
    alpha = 1
    for i in range(num_trees):
        L, y_l = bootstrap_sampler(x, y, x.shape[0])
        h_l = id3(L, y_l, max_depth=max_depth)
        # visualize(h_l)
        h_ens.append((alpha, h_l))
    return h_ens


def boosting(x, y, max_depth, num_stumps):
    """
    Implements an adaboost algorithm using the id3 algorithm as a base decision tree
    """
    h_ens = []
    num_samples = x.shape[0]
    w_l = np.ones(y.shape)
    w_l /= num_samples
    for i in range(num_stumps):
        # print("Weights: ", np.sum(w_l))
        h_l = id3(x, y, max_depth=max_depth, weights=w_l)
        y_pred = [predict_example(sample, h_l) for sample in x]
        eps_t = np.dot(np.absolute(y - y_pred), w_l)
        # print("Error: ", eps_t)
        alpha_t = 0.5 * np.log(((1 - eps_t) / eps_t))
        # print("Alpha: ", alpha_t)
        indicator = np.absolute(y - y_pred)
        for idx, w in enumerate(w_l):
            if indicator[idx]:
                w_l[idx] *= np.exp(alpha_t)
            else:
                w_l[idx] *= np.exp(-alpha_t)
        w_l /= 2 * np.sqrt(eps_t * (1 - eps_t))
        h_ens.append((alpha_t, h_l))
    return h_ens


def predict_example_ens(x, h_ens):
    """
    Predicts the classification label for a single example x using a combination of weighted trees
    Returns the predicted label of x according to tree
    """
    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    y_pred = sum([predict_example(x, h_l[1]) * h_l[0] for h_l in h_ens])
    y_pred /= sum([h_l[0] for h_l in h_ens])
    if y_pred > 0.5:
        y_pred = 1
    else:
        y_pred = 0
    return y_pred


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    return 1 / len(y_true) * (np.sum(y_true != y_pred))


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.
    Returns the predicted label of x according to tree
    """
    for decision_node, child_tree in tree.items():
        idx = decision_node[0]
        val = decision_node[1]
        decision = decision_node[2]

        if decision == (x[idx] == val):
            if type(child_tree) is not dict:
                class_label = child_tree
            else:
                class_label = predict_example(x, child_tree)

            return class_label


def confusion_matrix(y, y_pred, fig):
    confusionMatrix = np.zeros((2, 2))
    rows = ["Actual Positive", "Actual Negative"]
    cols = ("Classifier Positive", "Classifier Negative")
    for i, j in zip(y, y_pred):
        confusionMatrix[i][j] += 1
    temp = np.flip(confusionMatrix, 0)
    confusionMatrix = np.flip(temp, 1)
    fig.subplots_adjust(left=0.3, top=0.8, wspace=1)
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=2, rowspan=2)
    ax.table(cellText=confusionMatrix.tolist(),
             rowLabels=rows,
             colLabels=cols, loc="upper center")
    ax.axis("off")


if __name__ == '__main__':
    # Load the training data
    train_data = ['data/monks-1.train', 'data/monks-2.train', 'data/monks-3.train']
    test_data = ['data/monks-1.test', 'data/monks-2.test', 'data/monks-3.test']

    for i in range(3):
        M = np.genfromtxt(train_data[i], missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrn = M[:, 0]
        Xtrn = M[:, 1:]

        # Load the test data
        M = np.genfromtxt(test_data[i], missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytst = M[:, 0]
        Xtst = M[:, 1:]

        # Learn a decision tree of depth 3
        decision_tree = id3(Xtrn, ytrn, max_depth=3)
        visualize(decision_tree)

        # Compute the test error
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))

    """
    PART A - Four bagging models with maximum depth = 3, 5 and bag trees (k = 10, 20)
    """
    i = 1
    for depth in [3, 5]:
        for bootstrap in [10, 20]:
            h_ens_bag = bagging(Xtrn, ytrn, max_depth=depth, num_trees=bootstrap)
            y_pred_ens_bag = [predict_example_ens(x, h_ens_bag) for x in Xtst]
            tst_err_ens_bag = compute_error(ytst, y_pred_ens_bag)
            fig = plt.figure(i)
            confusion_matrix(ytst, y_pred_ens_bag, fig)
            fig.suptitle("Bagging: Max Depth " + str(depth) + " and Bag Size " + str(bootstrap))
            fig.savefig("Bagging_Figure_" + str(depth) + "_" + str(bootstrap) + ".png")
            i += 1

    """
    PART B - Four boosting models with max_depth = 1, 2 and num_stumps = 20, 40
    """
    for depth in [1, 2]:
        for stump in [20, 40]:
            h_ens_boost = boosting(Xtrn, ytrn, max_depth=depth, num_stumps=stump)
            y_pred_ens_boost = [predict_example_ens(x, h_ens_boost) for x in Xtst]
            tst_err_ens_boost = compute_error(ytst, y_pred_ens_boost)
            fig = plt.figure(i)
            confusion_matrix(ytst, y_pred_ens_boost, fig)
            fig.suptitle("Boosting: Max Depth " + str(depth) + " and Number of Stumps " + str(stump))
            fig.savefig("Boosting_Figure_" + str(depth) + "_" + str(stump) + ".png")
            i += 1

    """
    Part 3 scikit-learn
    """

    """
    Sklearn Bagging
    """
    for depth in [3, 5]:
        dtree = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
        for bootstrap in [10, 20]:
            sk_bag = BaggingClassifier(base_estimator=dtree, n_estimators=bootstrap)
            sk_bag.fit(Xtrn, ytrn)
            y_pred_sk_bag = sk_bag.predict(Xtst)
            tst_err_sk_bag = compute_error(ytst, y_pred_sk_bag)
            fig = plt.figure(i)
            confusion_matrix(ytst, y_pred_sk_bag, fig)
            fig.suptitle("Sklearn Bagging: Max Depth " + str(depth) + " and Bag Size " + str(bootstrap))
            fig.savefig("SkBagging_Figure_" + str(depth) + "_" + str(bootstrap) + ".png")
            i += 1

    """
    Sklearn Boosting
    """
    for depth in [1, 2]:
        dtree = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
        for stump in [20, 40]:
            sk_boost = AdaBoostClassifier(base_estimator=dtree, n_estimators=stump)
            sk_boost.fit(Xtrn, ytrn)
            y_pred_sk_boost = sk_boost.predict(Xtst)
            tst_err_ens_boost = compute_error(ytst, y_pred_sk_boost)
            fig = plt.figure(i)
            confusion_matrix(ytst, y_pred_sk_boost, fig)
            fig.suptitle("Sklearn Boosting: Max Depth " + str(depth) + " and Number of Stumps " + str(stump))
            fig.savefig("SkBoosting_Figure_" + str(depth) + "_" + str(stump) + ".png")
            i += 1

    plt.show()
