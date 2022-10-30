import pickle, os

import pandas as pd
from skimage.transform import rescale
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree

#flattening the images
def flatten_images():

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    return data, digits

#rescaling happening here
def preprocess(images, rescale_factor):

    resized_images = []
    for d in images:
        resized_images.append(rescale(d, rescale_factor, anti_aliasing = False))
    return resized_images


def create_splits(data, target, test_size, validation_size_from_test_size):

    # splits the data accordingly as values passing from the main
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=test_size, shuffle=False)

    X_test, X_validation, y_test, y_validation = train_test_split(
        X_test, y_test, test_size=validation_size_from_test_size, shuffle=False)
    
    return X_train, X_test, X_validation, y_train, y_test, y_validation


#training and saving the model
def train_and_save_model(classifier, hyperparameter_list, X_train, y_train, X_validation, y_validation):

    acc_val, f1_validation, model_location, non_skipped_values = [], [], [], []

    for hyperparameter_value in hyperparameter_list: 

        # Create a classifier: a support vector classifier
        if classifier == 'SVM':
            clf = svm.SVC()
            clf.set_params(**hyperparameter_value)
        # Create a classifier : for the decision tree
        elif classifier == 'DecisionTree':
            clf = tree.DecisionTreeClassifier(max_depth = hyperparameter_value)

        # Learn the digits on the train dataset
        clf.fit(X_train, y_train)

        # Predicting the value of the digits on the test dataset
        predicted_val = clf.predict(X_validation)

        val_accuracy_score = accuracy_score(y_validation, predicted_val)
        val_f1_score = f1_score(y_validation, predicted_val, average='weighted')

        if val_accuracy_score < 0.11:
            #print("Skipping for the hyperparameter value that accuracy is very less", hyperparameter_value)
            continue

        #Save the model to the mount
        saved_model = pickle.dumps(clf)
        model_path = '/Users/harshi/Desktop/IITJ/Sem_4/MLOPS/mlops-22/models'
        path = model_path + '/' + classifier + '_' + str(hyperparameter_value) + '.pkl' 
        with open(path, 'wb') as f:
            pickle.dump(clf, f)

        
        acc_val.append(val_accuracy_score)
        f1_validation.append(val_f1_score)
        model_location.append(path)
        non_skipped_values.append(hyperparameter_value)

    df = pd.DataFrame(data = {'Hyperparameter - Values ': non_skipped_values ,'Accuracy of Validation Data calculated': acc_val, 'f1 score of Validation Data calculated': f1_validation, 'Model Location stored': model_location})

    return df

# seperate method for returning the model path
def get_best_model_path(df):

    return df.iloc[df['Accuracy of Validation Data calculated'].argmax()]['Model Location stored']

# seperate method for returning the performance metrics
def get_best_model_metrics(df):

    return df.iloc[df['Accuracy of Validation Data calculated'].argmax()]['Accuracy of Validation Data calculated'], df.iloc[df['Accuracy of Validation Data calculated'].argmax()]['f1 score of Validation Data calculated']

# Predicting the test data using the best pickle model by selecting the best hyperparameters
def model_test(model_path, X, y):

    clf = pickle.load(open(model_path, 'rb'))

    predicted_test = clf.predict(X)

    accuracy_test = accuracy_score(y, predicted_test)
    f1_score_test = f1_score(y, predicted_test, average='weighted')
    
    return accuracy_test, f1_score_test


def print_data_frame (clf_name, df):

    print()
    print("Metrics calculated for:", clf_name)

def print_metrics(model_accuracy, model_f1_score):

    print()
    print("Accuracy on the Test Data:", model_accuracy)
    print("f1 score on the Test Data:", model_f1_score)
    print("###############################################################################################################")