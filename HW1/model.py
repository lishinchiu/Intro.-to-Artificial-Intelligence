import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class CarClassifier:
    def __init__(self, model_name, train_data, test_data):
        '''
        Convert the 'train_data' and 'test_data' into the format
        that can be used by scikit-learn models, and assign training images
        to self.x_train, training labels to self.y_train, testing images
        to self.x_test, and testing labels to self.y_test.These four 
        attributes will be used in 'train' method and 'eval' method.
        '''
        self.x_train, self.y_train, self.x_test, self.y_test = self.prepare_data(train_data, test_data)
        self.model = self.build_model(model_name)
        
    def prepare_data(self, train_data, test_data):
        '''
        Convert train_data and test_data into the format that can be used by scikit-learn models.
        Flatten the images and convert them into numpy arrays. 
        Assign the features and labels to self.x_train, self.y_train, self.x_test, self.y_test.
        '''
        x_train = np.array([item[0].flatten() for item in train_data])
        y_train = np.array([item[1] for item in train_data])
        x_test = np.array([item[0].flatten() for item in test_data])
        y_test = np.array([item[1] for item in test_data])
        return x_train, y_train, x_test, y_test
    
    def build_model(self, model_name):
        '''
        Build and return the correct model according to the 'model_name'.
        Supported models are KNN, Random Forest, and AdaBoost.
        '''
        # Begin your code (Part 2-2)
        if model_name == 'KNN':
            return KNeighborsClassifier(n_neighbors=3)
        elif model_name == 'RF':
            return RandomForestClassifier(n_estimators=150)
        elif model_name == 'AB':
            return AdaBoostClassifier(n_estimators=150)
        else:
            raise ValueError("Invalid model name. Supported models are: 'KNN', 'RF', 'AB'")
        # Begin your code (Part 2-2)
    def train(self):
        '''
        Fit the model on training data (self.x_train and self.y_train).
        '''
        # Begin your code (Part 2-3)
        self.model.fit(self.x_train, self.y_train)
        # Begin your code (Part 2-3)

    
    def eval(self):
        y_pred = self.model.predict(self.x_test)
        print(f"Accuracy: {round(accuracy_score(y_pred, self.y_test), 4)}")
        print("Confusion Matrix: ")
        print(confusion_matrix(y_pred, self.y_test))
    
    def classify(self, input):
        return self.model.predict(input.reshape(1, -1))[0]
