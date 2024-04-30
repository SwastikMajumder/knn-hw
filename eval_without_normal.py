import numpy as np
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def multi_dimension_euclid(x, y): # euclidean distance generalized to multiple dimensions
    number = 0
    arr = x-y
    for item in arr:
        number += item**2
    number = math.sqrt(number)
    return number

def scale_factor(content): # normalization
    min_c = np.min(content.reshape(-1))
    max_c = np.max(content.reshape(-1))
    return min_c, max_c

def classify(feature_info, unknown_point, k, category_info):
    sort_feature_info = []
    for i,item in enumerate(feature_info):
        sort_feature_info.append([multi_dimension_euclid(item, unknown_point),category_info[i]]) # appending category names with list for sorting
    sort_feature_info = sorted(sort_feature_info, key=lambda x: x[0]) # sort according to neighbour distance
    sort_feature_info = sort_feature_info[:k] # we consider only a few neighbors based on k value
    category = {} # name and occurence of neighbor categories
    for item in sort_feature_info:
        if item[1] not in category:
            category[item[1]] = 1 # create new category if it was not encountered before
        else:
            category[item[1]] += 1
    max_category = None
    max_occurence = -1 # how many times a category appear
    for key in category.keys():
        if category[key] > max_occurence: # the most occuring category
            max_occurence = category[key]
            max_category = key
    return max_category

for k in range(1,51+1,2): # k value
    print(k)
    avg_train_test = []
    avg_train_train = []
    
    for _ in range(20):
        data_set = None
        with open("iris.csv", "r") as file:
            data_set = file.read()
        data_set = data_set.split("\n")[:-2]
        data_set = np.array(data_set)
        data_set = shuffle(data_set)
        data_set = train_test_split(data_set, test_size=0.2)
        train, test = data_set
        train_f = np.array([np.array(item.split(",")[:-1]).astype(float) for item in train]) # the features
        test_f = np.array([np.array(item.split(",")[:-1]).astype(float) for item in test])
        train_c = [item.split(",")[-1] for item in train] # the category type
        test_c = [item.split(",")[-1] for item in test]
        # s = scale_factor(train_f)
        
        # min_c, max_c = s

        # test data evaluated on training data
        correct = 0
        for i in range(len(test_f)):    
            output = classify(train_f, test_f[i], k, train_c)
            if output == test_c[i]: # if correct category is predicted
                correct += 1
        avg_train_test.append(correct/len(test_c)) # accuracy score

        # training data evaluated on training data
        correct = 0
        for i in range(len(train_f)):    
            output = classify(train_f, train_f[i], k, train_c)
            if output == train_c[i]:
                correct += 1

        avg_train_train.append(correct/len(train_c))
    print(np.mean(np.array(avg_train_test))) # average accuracy over 20 times loop
    print(np.std(np.array(avg_train_test))) # for errors bars
    print(np.mean(np.array(avg_train_train)))
    print(np.std(np.array(avg_train_train)))
    print()
