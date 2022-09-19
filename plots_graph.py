# Standard scientific Python imports
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

# 1. set the ranges of hyper parameters 
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.5, 0.7, 1, 2] 

h_param_comb = [{'gamma':g, 'C':c} for g in gamma_list for c in c_list]

assert len(h_param_comb) == len(gamma_list)*len(c_list)

train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

#PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
print(" The original image size in the digits dataset is - ")
print(digits.images.shape)
print("------------------------------------------------------")
#PART: sanity check visualization of the data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

#PART: data pre-processing -- to remove some noise, to normalize data, format the data to be consumed by mode
# flatten the images
#n_samples = len(digits.images)
#data = digits.images.reshape((n_samples, -1))
#Resolution -1
#image_resized = resize(digits.images, (digits.images.shape[0],digits.images.shape[1] // 0.4, digits.images.shape[2] // 0.4),anti_aliasing=True)

#Resolution - 2
#image_resized = resize(digits.images, (digits.images.shape[0],digits.images.shape[1] //4, digits.images.shape[2] //4),anti_aliasing=True)
#Resolution - 3
image_resized = resize(digits.images, (digits.images.shape[0],digits.images.shape[1] //2, digits.images.shape[2] //2),anti_aliasing=True)
print(" The resized image is")
print(image_resized.shape)
print("------------------------------------------------------")
n_samples = len(image_resized)
data = digits.images.reshape((n_samples, -1))

#PART: define train/dev/test splits of experiment protocol
# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model
dev_test_frac = 1-train_frac
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
)


best_acc = -1.0
best_model = None
best_h_params = None
myTable = PrettyTable([ "The Hyperparameters taken" ,  "Train Accuracy", "Dev accuracy" , "Test accuracy" ])
# 2. For every combination-of-hyper-parameter values
for cur_h_params in h_param_comb:
    #print("| The Hyperparameters taken  |   train accuracy  | dev accuracy   | Test accuracy |")
    #PART: Define the model
    # Create a classifier: a support vector classifier
    clf = svm.SVC()

    #PART: setting up hyperparameter
    hyper_params = cur_h_params
    clf.set_params(**hyper_params)


    #PART: Train model
    # 2.a train the model 
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    train_acc = metrics.accuracy_score(y_pred=y_train_pred, y_true=y_train)
    #print("Training parameters :"+str(cur_h_params))
    #print("Training accuracy:" + str(train_acc))
    # print(cur_h_params)
    #PART: get dev set predictions
    predicted_dev = clf.predict(X_dev)

    # 2.b compute the accuracy on the validation set
    val_acc = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)

    # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest. 
    '''if cur_acc > best_acc:
        best_acc = cur_acc
        best_model = clf
        best_h_params = cur_h_params'''
    #print("Validation parameters :"+str(cur_h_params))
    #print("Validation accuracy:" + str(val_acc))

    predicted_test = clf.predict(X_test)
    Test_acc = metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)
    #print("Testing parameters :"+str(cur_h_params))
    #print("Testing accuracy:" + str(Test_acc))
    myTable.add_row([cur_h_params, round(train_acc,5),round(val_acc,5),  round(Test_acc,5)])
    #print("|", cur_h_params ,"|", round(train_acc,5) ,"|" , round(val_acc,2), "|" , round(Test_acc,2), "|")

    if val_acc > best_acc:
        best_acc = val_acc
        best_model = clf
        best_h_params = cur_h_params
print(myTable)
#PART: Get test set predictions
# Predict the value of the digit on the test subset
predicted_train = best_model.predict(X_train)
predicted_dev = best_model.predict(X_dev)
predicted_test = best_model.predict(X_test)

print("Best hyperparameters were:")
print(cur_h_params)
print("-----------------------------")
print("Training accuracy with the best hyperparameters", metrics.accuracy_score(y_pred=predicted_train, y_true=y_train))
print("Dev accuracy with the best hyperparameters", metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev))
print("Test accuracy with the best hyperparameters", metrics.accuracy_score(y_pred=predicted_test, y_true=y_test))

# 4. report the test set accurancy with that best model.
#PART: Compute evaluation metrics
print(
    f"Classification report for the Test classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted_test)}\n"
)
