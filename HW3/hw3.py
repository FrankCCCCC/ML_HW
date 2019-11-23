import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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

from mlxtend.plotting import plot_decision_regions

# Read Data
train = pd.read_csv('fashionmnist/fashion-mnist_train.csv')
test = pd.read_csv('fashionmnist/fashion-mnist_test.csv')
X_train = train.iloc[:, 1:].values
Y_train = train.iloc[:, 0].values
X_test = test.iloc[:, 1:].values
Y_test = test.iloc[:, 0].values
X_train_pca = np.ones(1)
X_test_pca = np.ones(1)
# print(X_train)

# PCA Components & Training Samples
comp_num = 400
train_sample = 30000

# Select Training Samples
def select_samples(x, y, train_sample):
    X_out = x[:train_sample, :]
    Y_out = y[:train_sample]
    return X_out, Y_out

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

def cross_validate(model, X_train, Y_train, cv_i):
    cv_scores = cross_val_score(estimator=model, X=X_train, y=Y_train, cv=cv_i, n_jobs=-1)
    cv_accuracy = np.mean(cv_scores)*100
    print("CV Accuracy Scores: ", cv_scores)
    print("CV Accuracy: ", cv_accuracy, "%")


X_train, Y_train = select_samples(X_train, Y_train, train_sample)
# X_train_pca, X_test_pca = pca(X_train, X_test)
# model = training(X_train_pca, Y_train)
# predict(model, X_test_pca, Y_test)

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
vc = VotingClassifier(estimators=clfs, voting='soft')
# bgcRnf = BaggingClassifier(base_estimator=rnf, n_estimators=3, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1, random_state=1)
# bgcKnn = BaggingClassifier(base_estimator=knn, n_estimators=3, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1, random_state=1)
# pipe_svm = make_pipeline(StandardScaler(), PCA(n_components=comp_num), svm)
# pipe_knn = make_pipeline(StandardScaler(), PCA(n_components=comp_num), knn)
# pipe_rnf = make_pipeline(StandardScaler(), PCA(n_components=comp_num), rnf)
# pipe_dct = make_pipeline(StandardScaler(), PCA(n_components=comp_num), dct)
# pipe_gnb = make_pipeline(StandardScaler(), PCA(n_components=comp_num), gnb)
pipe_mlp = make_pipeline(StandardScaler(), PCA(n_components=comp_num), mlp)
# pipe_bgcRnf = make_pipeline(StandardScaler(), PCA(n_components=comp_num), bgcRnf)
# pipe_bgcKnn = make_pipeline(StandardScaler(), PCA(n_components=comp_num), bgcKnn)
pipe_ovrMlp = make_pipeline(StandardScaler(), PCA(n_components=comp_num), ovr)
pipe_vc = make_pipeline(StandardScaler(), PCA(n_components=comp_num), vc)

model = pipe_vc

# cross_validate(model, X_train, Y_train, 10)

model.fit(X_train, Y_train)
test_accuracy(model, X_test, Y_test)
