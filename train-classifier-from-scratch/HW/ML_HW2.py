import numpy as np
# from sklearn import 
from urllib.request import urlretrieve
import pandas as pd
import matplotlib as plt
import csv

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def load_data(is_download = True):
    if is_download:
        urlretrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", "car.data")
        print("Downloaded car.data")
    col_features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    raw_data = pd.read_csv("car.data", names=col_features)
    print("Number of Data: ", len(raw_data))
    return raw_data

def split_test(dataframe, test_ratio=0.3):
    train = dataframe
    test = train.sample(frac=test_ratio, replace=True, random_state=1)
    train = train.drop(test.index)
    return train, test

def split_target(raw_dataframe):
    # dim = raw_dataframe.shape
    x = raw_dataframe.iloc[:, :-1]
    y = raw_dataframe.iloc[:, -1]
    # y.columns = ['class']
    y = pd.DataFrame(y, columns=['class'])
    return x, y

def group_target(targets_dataframe):

    return

def convert_onehot(dataframe):
    print('Converting Raw Data to Onehot Format')
    return pd.get_dummies(dataframe, prefix=dataframe.columns)

def label_encode(dataframe):
    encoder = LabelEncoder()
    y = encoder.fit_transform(dataframe.values)
    y = pd.DataFrame(y, columns=['class'])
    return encoder, y

def convert_numpy(data):
    if type(data) == type(pd.DataFrame(['ex', 0])):
        return data.to_numpy()
    if type(data) == type(np.array.zeros(1, 2)):
        return np.array(data)

def show_data():
    raw_data = load_data(True)
    onehot_data = convert_onehot(raw_data)
    for name in raw_data.keys():
        print(name, pd.unique(raw_data[name]))
    print(raw_data.head())
    print('\n')
    print(onehot_data.head())

def iris_test():
    iris = datasets.load_iris()
    print(type(iris.data))
    print(iris.data)
    print(iris.target)

def svm_classifier(data_x, data_y, kernel_i='rbf', is_plot_show = False):
    svm = SVC(kernel=kernel_i, C=1.0, gamma=0.45, random_state=1)
    svm.fit(data_x, data_y)
    return svm

def test_accuracy(classifier, test_x, test_y):
    predict_y = svm.predict(test_x)
    np_array = test_y.values.ravel()

    iters = 0
    error = 0
    for i in range(len(predict_y)):
        iters+=1
        a = np_array[i]
        p = predict_y[i]
        # print(a, " : ", p)
        if a != p:
            error+=1
    print("Accuacy: ", 100 - error/iters*100, '%')

raw_data = load_data()
x, y = split_target(raw_data)
onehot_x =  convert_onehot(x)
encoder, encoded_y = label_encode(y)
combine_data = pd.concat([onehot_x, encoded_y], axis=1, sort=False)
print(combine_data)
train, test = split_test(combine_data, 0.3)
train_x, train_y = split_target(train)
test_x, test_y = split_target(test)

# print(onehot_x.head())
# print(encoded_y.head())

svm = svm_classifier(train_x, train_y)
test_accuracy(svm, test_x, test_y)


# test = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21]], columns=['a', 'b', 'c'])
# print(test)
# s = test.sample(frac=0.3, replace=False, random_state=1)
# test = test.drop(s.index)
# print(test)
# print(s)
