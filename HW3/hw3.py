import numpy as np
from scipy import ndimage
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

from mlxtend.plotting import plot_decision_regions

path_data = 'fashionmnist/'
path_model = 'save/model/'
path_result = 'save/result/'

# Read Data
train = pd.read_csv(path_data + 'fashion-mnist_train.csv')
test = pd.read_csv(path_data + 'fashion-mnist_test.csv')
X_train = train.iloc[:, 1:].values
Y_train = train.iloc[:, 0].values
X_test = test.iloc[:, 1:].values
Y_test = test.iloc[:, 0].values
X_train_pca = np.ones(1)
X_test_pca = np.ones(1)
# print(X_train)

# PCA Components & Training Samples
comp_num = 400
train_sample = 1000

# svm = SVC(kernel='rbf', C=1.0, gamma=0.45, random_state=1)
# knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
# rnf = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1)
# dct = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=1)
# gnb = GaussianNB(priors=None)
# # nn_nodes = (400, 350, 300, 250, 200, 150, 100, 100, 100, 100)
# nn_nodes = (400, 500, 600, 700, 800, 700, 600, 500, 400, 200, 100, 50)
# # nn_nodes = (400, 350, 300, 250, 200, 150, 100)
# mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=nn_nodes, random_state=1)
# ovr = OneVsRestClassifier(mlp, n_jobs=-1)
# clfs = [('mlp', mlp), ('rnf', rnf), ('knn', knn), ('dct', dct)]
# vc = VotingClassifier(estimators=clfs, voting='soft')

# class Svm:
#     kernel = 'rbf'
#     C = 1.0
#     gamma = 0.45
#     random_state = 1
#     svm = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state)

# class Knn:
#     def __init__(self):
#         n_neighbors=5
#         p=2
#         metric='minkowski'

# class Rnf:
#     def __init__(self):
#         criterion='gini'
#         n_estimators=25
#         random_state=1


# Read Data
train = pd.read_csv(path_data + 'fashion-mnist_train.csv')
test = pd.read_csv(path_data + 'fashion-mnist_test.csv')
X_train = train.iloc[:, 1:].values
Y_train = train.iloc[:, 0].values
X_test = test.iloc[:, 1:].values
Y_test = test.iloc[:, 0].values
X_train_pca = np.ones(1)
X_test_pca = np.ones(1)
# print(X_train)

# PCA Components & Training Samples
comp_num = 400
train_sample = 1000

# Select Training Samples
def select_samples(x, y, train_sample):
    X_out = x[:train_sample, :]
    Y_out = y[:train_sample]
    return X_out, Y_out

def show_img(x):
    n_row = 1
    n_col = 5
    plt.figure(figsize=(10,8))
    for i in list(range(n_row*n_col)):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(x[i,:].reshape(28,28), cmap="gray")
        title_text = "Image" + str(i+1)
        plt.title(title_text, size=6.5)
    plt.show()


def pca(X_train, X_test):
    pca = PCA(n_components=comp_num)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    # print(pca.explained_variance_ratio_)
    l = len(pca.explained_variance_ratio_)
    print(l)
    plt.bar(range(1, comp_num + 1), pca.explained_variance_ratio_)
    plt.step(range(1, comp_num + 1), np.cumsum(pca.explained_variance_ratio_))
    plt.show()
    print(X_train_pca.shape)

    return X_train_pca, X_test_pca

def training(X_train_pca, Y_train):
    svm = SVC(kernel='rbf', C=1.0, gamma=0.45, random_state=1)
    model = OneVsRestClassifier(svm, n_jobs=-1)
    model.fit(X_train_pca, Y_train)
    return model

def predict(model, X_test_pca, Y_test):
    predict_y = model.predict(X_test_pca)
    targets = range(1, 11)
    classification_report(Y_test, predict_y, labels = targets)
    accuracy = np.mean(Y_test == predict_y) * 100
    print("Accuracy: ", accuracy, "%")

def train_score(pipe, x_train, y_train, x_test, y_test):
    pipe.fit(x_train, y_train)
    score = pipe.score(x_test, y_test)
    print("Accuracy: ", score*100, "%")
    return score

def test_accuracy(model, X_test, Y_test):
    model_predict = model.predict(X_test)
    model_predict = np.array(model_predict)
    test_set_accuracy = np.mean(Y_test == model_predict)*100
    print("Test Set Accuracy: ", test_set_accuracy, "%")
    return test_set_accuracy

def cross_validate(model, X_train, Y_train, cv_i):
    cv_scores = cross_val_score(estimator=model, X=X_train, y=Y_train, cv=cv_i, n_jobs=-1)
    cv_accuracy = np.mean(cv_scores)*100
    cv_std = np.std(cv_scores) * 100
    print("CV Accuracy Scores: ", cv_scores)
    print("CV Accuracy: ", cv_accuracy, ' +/- ' , cv_std, " %")
    return cv_scores, cv_accuracy

def convolution(x):
    fil_1 = lambda i: ndimage.gaussian_filter(i, sigma=1.4)
    # fil_2 = lambda i: ndimage.gaussian_gradient_magnitude(i, sigma=1)
    # fil_3 = lambda i: ndimage.percentile_filter(i, percentile=20, size=20)
    # fil_3 = lambda i: ndimage.rank_filter(i, rank=42, size=20)

    fil_kernel = lambda p: fil_1(p)
    conv = lambda i: fil_kernel(i.reshape(28, 28)).reshape(784)
    re = np.array([conv(i) for i in x])
    return re

def pca_data():
    pca(X_train, X_test)

# def write

def show_performance(model, model_name, X_train, Y_train, X_test, Y_test, is_write=False, is_cross_validate=False):
    if is_write:
        print('Saved Below')
    print('Model: ' + model_name)
    print('Test Condition: ')
    print('- ' + str(train_sample) + ' Samples')
    print('- Gaussian Filter Convolution Sigma = 1.4 ')
    print('- StandardScalar Normalize')
    print('- ' + str(comp_num) + ' PCA Components')
    if is_cross_validate:
        scores, avg_accuracy = cross_validate(model, X_train, Y_train, 5)
    model.fit(X_train, Y_train)
    accuracy = test_accuracy(model, X_test, Y_test)
    save_test_model(model, model_name, X_test, Y_test)
    if is_write:
        file_name = model_name + '.txt'
        file_path = path_result + file_name
        file = open(file_path, 'w')
        file.write('Model: ' + model_name + '\n')
        file.write('Test Condition: \n')
        file.write('- ' + str(train_sample) + ' Samples\n')
        file.write('- Gaussian Filter Convolution Sigma = 1.4\n')
        file.write('- StandardScalar Normalize\n')
        file.write('- ' + str(comp_num) + ' PCA Components\n')
        if is_cross_validate:
            file.write('Cross Validation Results (with 10 folds)\n')
            file.write('Scores: ' + str(np.array_str(scores)) + '\n')
            file.write('Average Accuracy: ' + str(avg_accuracy) + '%\n')
        file.write('Accuracy: ' + str(accuracy) + '%\n')
        file.close()

def save_test_model(model, name, X_test, Y_test):
    print('Saving Model...')
    joblib.dump(model, path_model + name)
    print('Loading Model...')
    clf_load = joblib.load(path_model + name)
    print('Test Saved Model')
    test_accuracy(clf_load, X_test, Y_test)
    print('\n')

def run_model(model, model_name, X_train, Y_train, X_test, Y_test, is_write=False, is_cross_validate=False):
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    show_performance(model, model_name+str(now), X_train, Y_train, X_test, Y_test, is_write, is_cross_validate)

svm = SVC(kernel='rbf', C=1.0, gamma=0.45, random_state=1)
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
rnf = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1)
dct = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=1)
gnb = GaussianNB(priors=None)
# nn_nodes = (400, 350, 300, 250, 200, 150, 100, 100, 100, 100)
nn_nodes = (400, 500, 600, 700, 800, 700, 600, 500, 400, 200, 100, 50)
# nn_nodes = (400, 350, 300, 250, 200, 150, 100)
mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=nn_nodes, random_state=1)
ovr = OneVsRestClassifier(mlp, n_jobs=-1)
clfs = [('mlp', mlp), ('rnf', rnf), ('knn', knn), ('dct', dct)]
vc = VotingClassifier(estimators=clfs, voting='soft', weights=[2, 1, 1.5, 1])
# bgcRnf = BaggingClassifier(base_estimator=rnf, n_estimators=3, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1, random_state=1)
# bgcKnn = BaggingClassifier(base_estimator=knn, n_estimators=3, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1, random_state=1)
# bgcMlp = BaggingClassifier(base_estimator=mlp, n_estimators=3, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1, random_state=1)
pipe_svm = make_pipeline(StandardScaler(), PCA(n_components=comp_num), svm)
pipe_knn = make_pipeline(StandardScaler(), PCA(n_components=comp_num), knn)
pipe_rnf = make_pipeline(StandardScaler(), PCA(n_components=comp_num), rnf)
pipe_dct = make_pipeline(StandardScaler(), PCA(n_components=comp_num), dct)
pipe_gnb = make_pipeline(StandardScaler(), PCA(n_components=comp_num), gnb)
pipe_mlp = make_pipeline(StandardScaler(), PCA(n_components=comp_num), mlp)
# pipe_bgcRnf = make_pipeline(StandardScaler(), PCA(n_components=comp_num), bgcRnf)
# pipe_bgcKnn = make_pipeline(StandardScaler(), PCA(n_components=comp_num), bgcKnn)
# pipe_ovrMlp = make_pipeline(StandardScaler(), PCA(n_components=comp_num), ovr)
pipe_vc = make_pipeline(StandardScaler(), PCA(n_components=comp_num), vc)

model = pipe_knn

X_train, Y_train = select_samples(X_train, Y_train, train_sample)
# show_img(X_train)
# show_img(X_test)
X_train = convolution(X_train)
X_test = convolution(X_test)

# cross_validate(model, X_train, Y_train, 10)
# model.fit(X_train, Y_train)
# test_accuracy(model, X_test, Y_test)
# save_test_model(model, 'clf_test', X_test, Y_test)

run_model(pipe_knn, 'clf_knn', X_train, Y_train, X_test, Y_test, True, True)
# run_model(pipe_rnf, 'clf_rnf', X_train, Y_train, X_test, Y_test, True, False)
# run_model(pipe_dct, 'clf_dct', X_train, Y_train, X_test, Y_test, True, False)
# run_model(pipe_mlp, 'clf_mlp', X_train, Y_train, X_test, Y_test, True, False)
# run_model(pipe_vc, 'clf_vc', X_train, Y_train, X_test, Y_test, True, False)
# run_model(pipe_svm, 'clf_snm', X_train, Y_train, X_test, Y_test, True, False)
# run_model(pipe_gnb, 'clf_gnb', X_train, Y_train, X_test, Y_test, True, False)