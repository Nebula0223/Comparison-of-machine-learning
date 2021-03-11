from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve
import imageio as io
import os

def ambiguous_acquire(data_index):
    res = np.argmin(np.abs(list(clf0.decision_function(X_pool.iloc[data_index]))))
    return data_index[res]

def SVM_print(clf, known_indexes, unknown_indexes, new_index=False, title=False, name=False):
    #SVM training part(already known)
    SVM_train_X = X_pool.iloc[known_indexes]
    SVM_train_Y = y_pool.iloc[known_indexes]
    #next to learn(most ambiguous and needs to be learned)
    SVM_wanted_X = X_pool.iloc[unknown_indexes]
    # if new_index is True, then SVM needs to be printed again
    if new_index:
        next_index = X_pool.iloc[new_index]
    next_a = clf.coef_[0, 0]
    next_b = clf.coef_[0, 1]
    next_c = clf.intercept_[0]
    if np.round(-(next_c / next_b), 2)<=0:
        next_equation = "y=" + str(np.round(-(next_a / next_b), 2)) + "x" + str(np.round(-(next_c / next_b), 2))
    else:
        next_equation = "y=" + str(np.round(-(next_a / next_b), 2)) + "x+" + str(np.round(-(next_c / next_b), 2))
    # to get the beginning part and the end part of X and Y, these will be used to print the line
    next_X_info = [min_X, max_X]
    next_Y_info = [-(next_a * min_X + next_c) / next_b, -(next_a * max_X + next_c) / next_b]

    '''
    the third printing(will be circulated for the prescribed times)
    update the SVM line if new index is offered
    '''
    fig = plt.figure(figsize=(10, 6))
    # print unknown dot
    plt.scatter(SVM_wanted_X[MOX_feature1], SVM_wanted_X[MOX_feature2], c='k', s=25, marker='.', label="unknown")
    # print known dot(CO concentration=0)
    plt.scatter(SVM_train_X[MOX_feature1][SVM_train_Y == 0], SVM_train_X[MOX_feature2][SVM_train_Y == 0], c='r', s=25, marker='.', label="CO concentration=0")
    # print known dot(CO concentration=20 ppm)
    plt.scatter(SVM_train_X[MOX_feature1][SVM_train_Y == 1], SVM_train_X[MOX_feature2][SVM_train_Y == 1], c='c', s=25, marker='.', label="CO concentration=20")
    # print new SVM line(based on a few data, might inaccurate)
    plt.plot(next_X_info, next_Y_info, c='m', label=next_equation, linewidth=3)
    # print original SVM line(based on the full data, accurate)
    plt.plot(original_X_info, original_Y_info, '--', c='b', label=equation, linewidth=3)
    # print the most ambiguous dot
    if new_index:
        plt.scatter(next_index[MOX_feature1], next_index[MOX_feature2], c='gold', s=25, marker='*', label="next to learn")
    if title:
        plt.title(title,fontsize="xx-large",fontweight="bold")
    plt.xlabel(MOX_feature1,fontsize="x-large")
    plt.ylabel(MOX_feature2,fontsize="x-large")
    plt.legend(fontsize="large",markerscale=3,loc="upper right")
    if name:
        fig.set_size_inches((10, 6))
        plt.savefig(name, dpi=100)
    plt.show()
    return


MOX_path="./MOX Conclusion.csv"
MOX_data=pd.read_csv(MOX_path)
# print the scatterplot with the factor of R12_Down_Slope and R14_Up_Slope
# So these two factor will be used to identify the CO concentration(0 or 20?)
MOX_feature1="R12_Down_Slope"
MOX_feature2="R14_Up_Slope"
MOX_features=[MOX_feature1,MOX_feature2]
X=MOX_data[MOX_features] # X as feature
Y=MOX_data.CO_Concentration # Y as prediction target
# choose to identify the CO concentration(0 or 20?)
Y[Y==0] = 0
Y[Y==20] = 1
# CO_0 represents the CO concentration of 0, and CO_20 represents the CO concentration of 20 ppm
CO_0 = (Y==0)
CO_20 = (Y==1)
# CO_0 and CO_20 correspond to the Y of 0 and 1(which are both smaller than 2)
# Thus I use "<2" to keep the data I need and exclude the others
X1 = X[Y<2]
Y1 = Y[Y<2]
# to set the index again
X1 = X1.reset_index(drop=True)
Y1 = Y1.reset_index(drop=True)
'''
the first printing
I will print the scatterplot directly with the original data
'''
fig1=plt.figure(figsize=[10,6])
plt.title("Scatterplot of CO Concentration",fontsize="xx-large",fontweight="bold")
plt.scatter(X[MOX_feature1][CO_0], X[MOX_feature2][CO_0], c='r',s=25,marker=".",label="CO concentration=0")
plt.scatter(X[MOX_feature1][CO_20], X[MOX_feature2][CO_20],c='g',s=25,marker=".",label="CO concentration=20")
plt.xlabel(MOX_feature1,fontsize="x-large")
plt.ylabel(MOX_feature2,fontsize="x-large")
plt.legend(fontsize="large",markerscale=3,loc="upper right")
fig1.set_size_inches((10, 6))
plt.savefig("./scatter.png", dpi=100)
plt.show()

Y1 = Y1.astype(dtype=np.uint8) # to change the type of Y1
# to establish a model with the use of SVM, and acquire the information of the SVM line
clf0 = LinearSVC()
clf0.fit(X1, Y1)
# to get the min X and the max X
min_X = X1[MOX_feature1].min()
max_X = X1[MOX_feature1].max()
# to get the equation of the SVM line
# note that the following a,b,c represent that a*x + b*y + c = 0
# thus the equation is y = -(a/b)x -(c/b)
a = clf0.coef_[0, 0]
b = clf0.coef_[0, 1]
c = clf0.intercept_[0]
if np.round(-(c/b),2)<=0:
    equation="y="+str(np.round(-(a/b),2))+"x"+str(np.round(-(c/b),2))
else:
    equation = "y=" + str(np.round(-(a / b), 2)) + "x+" + str(np.round(-(c / b), 2))
# to get the beginning part and the end part of X and Y, these will be used to print the line
original_X_info = [min_X,max_X]
original_Y_info = [-(a*min_X + c)/b,-(a*max_X + c)/b]

'''
the second printing
the SVM line will be added to the base of the scatterplot
so that I can divide two kinds of CO concentration by analyzing the R12_Down_Slope and the R14_Up_Slope
'''
fig2=plt.figure(figsize=[10,6])
plt.title("Scatterplot with SVM Line",fontsize="xx-large",fontweight="bold")
# print the scatterplot part
plt.scatter(X1[MOX_feature1][Y1==0], X1[MOX_feature2][Y1==0], c='r',s=25,marker=".",label="CO concentration=0")
plt.scatter(X1[MOX_feature1][Y1==1], X1[MOX_feature2][Y1==1], c='g',s=25,marker=".",label="CO concentration=20")
# print the SVM line
plt.plot(original_X_info, original_Y_info, c='b',label=equation,linewidth=3)
plt.xlabel(MOX_feature1,fontsize="x-large")
plt.ylabel(MOX_feature2,fontsize="x-large")
plt.legend(fontsize="large",markerscale=3,loc="upper right")
fig2.set_size_inches((10, 6))
plt.savefig("./scatter with line.png", dpi=100)
plt.show()

X_pool, X_test, y_pool, y_test = train_test_split(X1, Y1, test_size=0.2, random_state=1)
# to set the index again
X_pool, X_test, y_pool, y_test = X_pool.reset_index(drop=True), X_test.reset_index(drop=True), y_pool.reset_index(drop=True), y_test.reset_index(drop=True)


known_indexes = list(range(40))
unknown_indexes = list(range(40, 150))
train_X = X_pool.iloc[known_indexes]
train_Y = y_pool.iloc[known_indexes]
clf = LinearSVC()
clf.fit(train_X, train_Y)
# create a folder to store the images of active learning
img_folder = "active_learning_image/"
try:
    os.mkdir(img_folder)
except:
    pass
filenames = []

title = "Initial Situation"
name = img_folder + ("activeLearning_initial.jpg")
SVM_print(clf, known_indexes, unknown_indexes, False, title, name)
filenames.append(name)
# calculate which index is most wanted(most ambiguous), and remove it from unknown index
new_index = ambiguous_acquire(unknown_indexes)
unknown_indexes.remove(new_index)

title = "ActiveLearning Times: 0"
name = img_folder + ("activeLearning_img0.jpg")
SVM_print(clf, known_indexes, unknown_indexes, new_index, title, name)
filenames.append(name)
for i in range(30):
    # move next_to_learn dot to known_indexes
    known_indexes.append(new_index)
    train_X = X_pool.iloc[known_indexes]
    train_Y = y_pool.iloc[known_indexes]
    # to fit the line with the new known_indexes
    clf = LinearSVC()
    clf.fit(train_X, train_Y)
    # print a new SVM image
    title = "ActiveLearning Times: %d" % (i+1)
    name = img_folder + ("activeLearning_img%d.jpg" % (i+1))
    new_index = ambiguous_acquire(unknown_indexes)
    unknown_indexes.remove(new_index)
    SVM_print(clf, known_indexes, unknown_indexes, new_index, title, name)
    filenames.append(name)
    # create a gif file with the image painted before
    images = []
    for filename in filenames:
        images.append(io.imread(filename))
    io.mimsave('activeLearning.gif', images, duration=1)
plt.show()

# print the learning curve for active learning
fig=plt.figure(figsize=[10,6])
plt.title("Learning Curve",fontsize="xx-large",fontweight="bold")
plt.ylim((0.5,0.8))
plt.xlabel("Number of training samples",fontsize="x-large")
plt.ylabel("Accuracy",fontsize="x-large")
train_sizes, train_scores, test_scores = learning_curve(SVC(gamma=0.05), X1, Y1, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=1))
#acquire the mean and standard deviation of the train accuracy and the test accuracy
train_accuracy_mean = np.mean(train_scores, axis=1)
train_accuracy_std = np.std(train_scores, axis=1)
test_accuracy_mean = np.mean(test_scores, axis=1)
test_accuracy_std = np.std(test_scores, axis=1)
print("active learning mean: ",test_accuracy_mean)
print("active learning std: ",test_accuracy_std)
plt.grid()
#plt.plot(train_sizes, train_accuracy_mean, 'o-', color="g",label="Training accuracy")
plt.plot(train_sizes, test_accuracy_mean, 'o-', color="b",label="Active Learning")

# print the learning curve for random sampling
train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), X1, Y1, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=1))
#acquire the mean and standard deviation of the train accuracy and the test accuracy
print(train_sizes)
train_accuracy_mean = np.mean(train_scores, axis=1)
train_accuracy_std = np.std(train_scores, axis=1)
test_accuracy_mean = np.mean(test_scores, axis=1)
test_accuracy_std = np.std(test_scores, axis=1)
print("random sampling mean: ",test_accuracy_mean)
print("random sampling std: ",test_accuracy_std)
#plt.plot(train_sizes, train_accuracy_mean, 'o-', color="g",label="Training accuracy")
plt.plot(train_sizes, test_accuracy_mean, 'o-', color="g",label="Random Sampling")
plt.legend(fontsize="large",markerscale=1,loc="upper right")
fig.set_size_inches((10, 6))
plt.savefig("learning curve", dpi=100)
plt.show()