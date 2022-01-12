import numpy as np
from numpy.lib.scimath import log2
from numpy.random import default_rng

""" Use loadtxt function to load the dataset from the given path.
    Args:
        filepath: the path of the .txt file to be loaded
    Returns:
        A dataset extracted from the file
"""
def read_dataset(filepath):
    dataset = np.loadtxt(filepath)
    return dataset

""" Split the original dataset into two arrays.
    Args:
        dataset: the input dataset
    Returns:
        tuples: (data_pair[0], data_pair[1], classes)
            data_pair[0]: dataset[:,:7] (attributes) part 
            data_pair[1]: dataset[:,7] (labels) part
            classes: unique labels in the dataset
"""
def split_dataset(dataset):
    data_pair = np.split(dataset, [7], axis=1)
    classes = np.unique(data_pair[1])
    return (data_pair[0], data_pair[1], classes)

""" Follow the given formula to get the entropy of a dataset.
    Args:
        labels(arrays): the input labels
        classes(arrays): the unique labels of the "labels"
    Returns:
        h: the final entropy of the dataset
"""
def entropy(labels, classes):
    h = 0.
    for class_ in classes:
        pk = np.count_nonzero(labels[labels == class_]) / np.count_nonzero(labels)
        res = pk * log2(pk)
        h = h - res
    return h

""" Calculate the information gain by the given formula.
    Args:
        s_all: the total dataset
        s_left: the left dataset
        s_right: the right dataset
    Returns:
        h: the final information gain
"""
def information_gain(s_all, s_left, s_right):
    (_, all_labels, all_classes) = split_dataset(s_all)
    (_, left_labels, left_classes) = split_dataset(s_left)
    (_, right_labels, right_classes) = split_dataset(s_right)
    h_all = entropy(all_labels, all_classes)
    h_left = entropy(left_labels, left_classes)
    h_right = entropy(right_labels, right_classes)
    cnt_left = np.count_nonzero(s_left)
    cnt_right = np.count_nonzero(s_right)
    cnt_all = cnt_left + cnt_right
    remainder = (cnt_left / cnt_all) * h_left + (cnt_right / cnt_all) * h_right
    info_gain = h_all - remainder
    return info_gain

""" Given a dataset, sort it for every columns at first, then try to find the best spilt point.
    A split point appears when there is an difference between two attributes, then we calculate 
    the information gain, the best split point will be the one with largest information gain.
    Args:
        dataset: the given dataset
    Returns:
        tuple: (final_attr, final_split_point)
            final_attr: the attribute of the final split point
            final_split_point: the split point value
"""
def find_split(dataset):
    (attributes, labels, _) = split_dataset(dataset)
    sorted_indeces = np.argsort(attributes, axis=0)
    sample_num = len(labels[:,0])
    final_split_point = 0.
    final_info_gain = 0
    final_attr = -1

    for i in range(len(attributes[0])):
        attribute = attributes[:,i]
        if(len(np.unique(attribute)) != 1):            
            sorted_attribute = attribute[sorted_indeces[:,i]]
            sorted_dataset = dataset[sorted_indeces[:,i]]
            for j in range(sample_num - 1):
                if(sorted_attribute[j] != sorted_attribute[j + 1]):
                    split_point = (sorted_attribute[j] + sorted_attribute[j + 1]) / 2
                    s_all = sorted_dataset
                    s_left = sorted_dataset[:j+1,]
                    s_right = sorted_dataset[j+1:,]
                    info_gain = information_gain(s_all, s_left, s_right)

                    if(info_gain > final_info_gain):
                        final_info_gain= info_gain
                        final_split_point = split_point
                        final_attr = i     

    return (final_attr, final_split_point)

""" Given a training dataset, convert it into a decision tree.
    Every time we find the best split point of the current dataset and split it into left and right parts,
    then recursively fo the same thing for these two parts until we get a leaf node. 
    Then we combine the left node and right node recursively to make a final decision tree.
    Args:
        training_dataset: the dataset need to be trained
        depth: the depth
    Returns:
        tuple: (node, depth)
            node: the current node
            depth: the depth from the current node to the deepest leaves
"""
def decision_tree_learning(training_dataset, depth):

    labels = training_dataset[:,-1]  
    unique_labels = np.unique(labels)

    if(len(unique_labels) == 1):
        return ({'attribute': None,
                'value': unique_labels[0],
                'left': None,
                'right': None,
                'is_leaf': True}, depth)
    else:
        (split_attr, split_point) = find_split(training_dataset)
        left = np.array([x for x in training_dataset if x[split_attr] < split_point])

        if(len(left) != 0):
            (left_node, l_depth) = decision_tree_learning(left, depth + 1)
        else:
            left_node = None

        right = np.array([x for x in training_dataset if x[split_attr] >= split_point])
        if(len(right) != 0):
            (right_node,r_depth) = decision_tree_learning(right, depth + 1)
        else:
            right_node = None

        node = {'attribute': split_attr,
                'value': split_point,
                'left': left_node,
                'right': right_node,
                'is_leaf': False}
        return (node, max(l_depth, r_depth))

# The following are for step3 -----------------------------------------------

""" Evaluate the trained tree by calculating a given test dataset's accuarcy.
    Args:
        test_db: the test dataset
        trained_tree: the trained dataset
    Returns:
        tuple: (accuracy, predictions)
            accuracy: the accuracy of the test dataset
            predictions: the prediction labels of the test dataset
"""
def evaluate(test_db, trained_tree):
    predictions = np.zeros((len(test_db), ), dtype=test_db.dtype)
    for (i, instance) in enumerate(test_db):
        node = trained_tree
        while(not node.get('is_leaf')):
            if(instance[node.get('attribute')] < node.get('value')):
                node = node.get('left')
            else:
                node = node.get('right')
        predictions[i] = node.get('value')
    
    golds = test_db[:,7]
    try:
        return (np.sum(golds == predictions) / len(golds), predictions)
    except ZeroDivisionError:
        return (0., predictions)

""" Evaluate the training dataset by "K folds cross validation"
    Args:
        dataset: the training dataset
        splits_num: the K, the number of folds
        need_be_pruned: if it is true, then we need to prune the decision tree we built every time
    Returns:
        tuple: (final_accuracy, final_recalls, final_precisions, final_f1scores, final_matrix, avg_depth)
"""
def cross_validation(dataset, splits_num, need_be_pruned):
    random_generator = default_rng()
    shuffled_indices = random_generator.permutation(len(dataset))
    split_indices = np.array_split(shuffled_indices, splits_num)

    folds = []
    for n in range(splits_num):
        test_indices = split_indices[n]
        train_indices = np.hstack(split_indices[:n] + split_indices[n+1:])
        folds.append([train_indices, test_indices])

    (_, _, classes) = split_dataset(dataset)

    folds_accuracies = np.zeros((splits_num, ), dtype=dataset.dtype)
    folds_precisions = np.zeros((splits_num, len(classes)), dtype=dataset.dtype)
    folds_recalls = np.zeros((splits_num, len(classes)), dtype=dataset.dtype)
    folds_f1scores = np.zeros((splits_num, len(classes)), dtype=dataset.dtype)
    folds_matrices_sum = np.zeros((len(classes), len(classes)), dtype=dataset.dtype)
    depth_sum = 0

    for i, (train_indices, test_indices) in enumerate(folds):
        training_dataset = dataset[train_indices]
        test_dataset = dataset[test_indices]

        #the '_' here represents the depth
        (root, depth) = decision_tree_learning(training_dataset, 0)
        depth_sum += depth

        if (need_be_pruned):
            root = prune_tree(root, training_dataset, test_dataset)

        acc, predictions = evaluate(test_dataset, root)
        folds_accuracies[i] = acc

        # The following are for calculating the required metrics --------------------
        golds = test_dataset[:,7]
        matrix = confusion_matrix(golds, predictions)
        folds_matrices_sum += matrix

        recalls = np.zeros((len(matrix), ))
        for c in range(matrix.shape[0]):
            if np.sum(matrix[c, :]) > 0:
                recalls[c] = matrix[c, c] / np.sum(matrix[c, :])
        folds_recalls[i] = recalls

        precisions = np.zeros((len(matrix), ))
        for c in range(matrix.shape[0]):
            if np.sum(matrix[:, c]) > 0:
                precisions[c] = matrix[c, c] / np.sum(matrix[:, c])
        folds_precisions[i] = precisions

        assert len(precisions) == len(recalls)
        f = np.zeros((len(precisions), ))
        for c, (p, r) in enumerate(zip(precisions, recalls)):
            if p + r > 0:
                f[c] = 2 * p * r / (p + r)
        folds_f1scores[i] = f
    
    final_accuracy = np.mean(folds_accuracies)
    final_recalls = np.mean(folds_recalls, axis=0)
    final_precisions = np.mean(folds_precisions, axis=0)
    final_f1scores = np.mean(folds_f1scores, axis=0)
    final_matrix = folds_matrices_sum / splits_num
    avg_depth = depth_sum / splits_num

    return (final_accuracy, final_recalls, final_precisions, final_f1scores, final_matrix, avg_depth)

""" Get a confusion matrix.
    Args:
        y_gold: the correct labels
        y_prediction: the cprediction labels
    Returns:
        confusion: a two-deminsional array represent the confusion matrix
"""
def confusion_matrix(y_gold, y_prediction):
    class_labels = np.unique(np.concatenate((y_gold, y_prediction)))
    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=int)
    for (i, label) in enumerate(class_labels):
        indices = (y_gold == label)
        predictions = y_prediction[indices]
        (unique_labels, counts) = np.unique(predictions, return_counts=True)

        frequency_dict = dict(zip(unique_labels, counts))

        for (j, class_label) in enumerate(class_labels):
            confusion[i, j] = frequency_dict.get(class_label, 0)

    return confusion

# The following are for step4 ---------------------------------------------

""" Given an array of labels and the current leaf's label, count the number of mismatches.
    Args:
        validation(array): the validation labels
        leaf: the label of the current leaf
    Returns:
        error_count: the total number of errors
"""
def count_vali_errors(validation, leaf):
    error_count = 0
    for data in validation:
        if data[7] != leaf:
            error_count += 1
    return error_count

""" Split a dataset into two by the given attribute and split value
    Args:
        dataset: the given dataset
        split_attr: the given split attribute
        split_point: the split point
    Returns:
        tuple: (left, right)
            left: left subset
            right: right subset
"""
def split_dataset_by_split_point(dataset, split_attr, split_point):
    left = np.array([x for x in dataset if x[split_attr] < split_point])
    right = np.array([x for x in dataset if x[split_attr] >= split_point])
    return (left, right)

""" Given a training dataset, return the label which has the maximum count in the dataset.
    This is used for find the most possible label for the new leaf after pruning.
    Args:
        dataset: the given training dataset
    Returns:
        curr_maximum_label: the label with the maximum number
"""
def find_maximum_label(dataset):
    label_index = len(dataset[0]) - 1
    unique_labels = np.unique(dataset[:, label_index])
    curr_maximum_label = 0
    curr_maximum_count = 0
    for label in unique_labels:
        count = len([x for x in dataset if x[label_index] == label])
        if (count > curr_maximum_count):
            curr_maximum_label = label
            curr_maximum_count = count
    return curr_maximum_label

""" Check whether pruning leaves will reduce validation errors, if it does, then return a new node with the new label
    Args:
        left_node: the left node of the current node
        right_node: the right node of the current node
        left_validation: the validation dataset belong to the left node
        right_validation: the validation dataset belong to the right node
        validation_dataset: the validation dataset
        training_dataset: the training dataset of the current node
    Returns:
        new_node: the new node after pruning (may do not need to prune which means unchnaged)
"""
def check_prune(left_node, right_node, left_validation, right_validation, validation_dataset, training_dataset):
            
    left_label = left_node.get('value')
    right_label = right_node.get('value')

    left_vali_errors = count_vali_errors(left_validation, left_label)
    right_vali_errors = count_vali_errors(right_validation, right_label)
    # The total number of validation errors before pruning the leaves
    previous_vali_errors = left_vali_errors + right_vali_errors

    # Find the label the new leaf node should be after pruning
    new_label = find_maximum_label(training_dataset)

    # The total number of validation errors after pruning the leaves
    after_pruning_vali_errors = count_vali_errors(validation_dataset, new_label)

    # Check whether pruning leaves will reduce validation errors
    if (after_pruning_vali_errors < previous_vali_errors):
        # In this case we need to prune
        new_node = {'attribute': None,
                    'value': new_label,
                    'left': None,
                    'right': None,
                    'is_leaf': True}
        return new_node
    else:
        # In this case we don't need to prune
        return None

""" Prune the trained model to reduce validation errors.
    Every time we need to check whether the current node meets the condition to be pruned.
    The first condition is that the left node and the right node of the current node are both leaves.
    The second condition is that it must reduce validation errors after pruning, otherwise we shouldn't prune it.
    In addition, after merging the left pruned node and the right pruned node, still need to check whether we can
        prune the new node.
    Args:
        node: the current node
        training_dataset: the training dataset
        validation_dataset: the validation dataset
    Returns:
        new_node: the new node after pruning (may do not need to prune which means unchnaged)
"""
def prune_tree(node, training_dataset, validation_dataset):

    if node.get('is_leaf'):
        return node
    
    left_node = node.get('left')
    right_node = node.get('right')
    if left_node is None:
        return prune_tree(right_node, training_dataset, validation_dataset)
    elif right_node is None:
        return prune_tree(left_node, training_dataset, validation_dataset)
    else:
        curr_attribute = node.get('attribute')
        curr_split_point = node.get('value')
        (left_training, right_training) = split_dataset_by_split_point(training_dataset, curr_attribute, curr_split_point)
        (left_validation, right_validation) = split_dataset_by_split_point(validation_dataset, curr_attribute, curr_split_point)

        if left_node.get('is_leaf') and right_node.get('is_leaf'):
            # In this case the tree may need to be pruned
            pruned_node = check_prune(left_node, right_node, left_validation, right_validation, validation_dataset, training_dataset)
            if pruned_node is None:
                return node
            else:
                return pruned_node
        
        else:
            new_left_node = prune_tree(left_node, left_training, left_validation)
            new_right_node = prune_tree(right_node, right_training, right_validation)
            new_node = {'attribute': curr_attribute,
                        'value': curr_split_point,
                        'left': new_left_node,
                        'right': new_right_node,
                        'is_leaf': False}

            # Still need to check whether we can prune the new node after we merging
            if new_left_node.get('is_leaf') and new_right_node.get('is_leaf'):
                pruned_node = check_prune(new_left_node, new_right_node, left_validation, right_validation, validation_dataset, training_dataset)
                if pruned_node is not None:
                    new_node = pruned_node

            return new_node

""" Split the indices into k folds, every time choose a fold as the test set and the remaining as the train+validation set.
    Args:
        split_indices(array): the indices array to be split
        splits_num: the number of folds
    Returns:
        folds: the final result consist of K different split solutions
"""
def split_into_k_folds(split_indices, splits_num):
    folds = []
    for n in range(splits_num):
        test_indices = split_indices[n]
        trainval_indices = np.hstack(split_indices[:n] + split_indices[n+1:])
        folds.append([trainval_indices, test_indices])
    return folds

""" Evaluate the training dataset by "K folds nested cross validation"
    Args:
        dataset: the training dataset
        n_outer_folds: the number of outer folds
        n_inner_folds: the number of inner folds
    Returns:
        tuple: (final_accuracy, final_recalls, final_precisions, final_f1scores, final_matrix, avg_depth)
"""
def nested_cross_validation(dataset, n_outer_folds, n_inner_folds):
    
    random_generator = default_rng()
    shuffled_indices = random_generator.permutation(len(dataset))
    split_indices = np.array_split(shuffled_indices, n_outer_folds)

    folds = split_into_k_folds(split_indices, n_outer_folds)

    (_, _, classes) = split_dataset(dataset)

    folds_accuracies = np.zeros((n_outer_folds, ), dtype=dataset.dtype)
    folds_precisions = np.zeros((n_outer_folds, len(classes)), dtype=dataset.dtype)
    folds_recalls = np.zeros((n_outer_folds, len(classes)), dtype=dataset.dtype)
    folds_f1scores = np.zeros((n_outer_folds, len(classes)), dtype=dataset.dtype)
    folds_matrices_sum = np.zeros((len(classes), len(classes)), dtype=dataset.dtype)
    depth_sum = 0

    for i, (train_indices, _) in enumerate(folds):

        trainval_dataset = dataset[train_indices]

        need_be_pruned = True
        (accuracy, recalls, precisions, f1scores, matrix, depth) = cross_validation(trainval_dataset, n_inner_folds, need_be_pruned)

        folds_accuracies[i] = accuracy
        folds_precisions[i] = precisions
        folds_recalls[i] = recalls
        folds_f1scores[i] = f1scores
        folds_matrices_sum += matrix
        depth_sum += depth

    final_accuracy = np.mean(folds_accuracies)
    final_recalls = np.mean(folds_recalls, axis=0)
    final_precisions = np.mean(folds_precisions, axis=0)
    final_f1scores = np.mean(folds_f1scores, axis=0)
    final_matrix = folds_matrices_sum / n_outer_folds
    avg_depth = depth_sum / n_outer_folds

    return (final_accuracy, final_recalls, final_precisions, final_f1scores, final_matrix, avg_depth)

clean_dataset = read_dataset("./wifi_db/clean_dataset.txt")
noisy_dataset = read_dataset("./wifi_db/noisy_dataset.txt")

print("----------------------------------------------------------------clean_dataset-----------------------------------")

(final_accuracy, final_recalls, final_precisions, final_f1scores, final_matrix, clean_avg_depth) = cross_validation(clean_dataset, 10, False)

print('cross_validation_final_accuracy:', final_accuracy)
print('cross_validation_final_recalls:', final_recalls)
print('cross_validation_final_precisions:', final_precisions)
print('cross_validation_final_f1score:', final_f1scores)
print('cross_validation_final_confusion_matrix:', final_matrix)


(nested_final_accuracy, nested_final_recalls, nested_final_precisions, nested_final_f1scores, nested_final_matrix, nested_clean_avg_depth) = nested_cross_validation(clean_dataset, 10, 9)
print('nested_cross_validation_final_accuracy:', nested_final_accuracy)
print('nested_cross_validation_final_recalls:', nested_final_recalls)
print('nested_cross_validation_final_precisions:', nested_final_precisions)
print('nested_cross_validation_final_f1score:', nested_final_f1scores)
print('nested_cross_validation_final_confusion_matrix:', nested_final_matrix)

print("-----------------------------------------------------------------------noisy_datasaet----------------------------")

(final_accuracy, final_recalls, final_precisions, final_f1scores, final_matrix, noisy_avg_depth) = cross_validation(noisy_dataset, 10, False)

print('cross_validation_final_accuracy:', final_accuracy)
print('cross_validation_final_recalls:', final_recalls)
print('cross_validation_final_precisions:', final_precisions)
print('cross_validation_final_f1score:', final_f1scores)
print('cross_validation_final_confusion_matrix:', final_matrix)


(nested_final_accuracy, nested_final_recalls, nested_final_precisions, nested_final_f1scores, nested_final_matrix, nested_noisy_avg_depth) = nested_cross_validation(noisy_dataset, 10, 9)
print('nested_cross_validation_final_accuracy:', nested_final_accuracy)
print('nested_cross_validation_final_recalls:', nested_final_recalls)
print('nested_cross_validation_final_precisions:', nested_final_precisions)
print('nested_cross_validation_final_f1score:', nested_final_f1scores)
print('nested_cross_validation_final_confusion_matrix:', nested_final_matrix)

print("-----------------------------------------------------depth analysis----------------------------")

print(">>>>>>>>>>>>>>>>>>>> before pruning: ")
print("the average depth of the clean dataset: ", clean_avg_depth)
print("the average depth of the noisy dataset: ", noisy_avg_depth)

print(">>>>>>>>>>>>>>>>>>>> after pruning: ")
print("the average depth of the clean dataset: ", nested_clean_avg_depth)
print("the average depth of the noisy dataset: ", nested_noisy_avg_depth)

