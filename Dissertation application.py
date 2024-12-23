import pandas
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

# Data definitions - secondary data
# timestamp	
# pCut::Motor_Torque	
# pCut::CTRL_Position_controller::Lag_error	
# pCut::CTRL_Position_controller::Actual_position	
# pCut::CTRL_Position_controller::Actual_speed	
# pSvolFilm::CTRL_Position_controller::Actual_position	
# pSvolFilm::CTRL_Position_controller::Actual_speed	
# pSvolFilm::CTRL_Position_controller::Lag_error	
# pSpintor::VAX_speed
# Replaced blade --New column added by me. This indicates a 0 for old blade and 1 for new blade

### Import training data ###
print("Which training dataset would you like to use?")
print("1.   training_data_mixed_blade_half1 (NewBlade001, OldBlade001)")
print("2.   training_data_mixed_blade_half2 (NewBlade003, OldBlade003)")
print("3.   training_data_mixed_blade (NewBlade001, NewBlade003, OldBlade001, OldBlade003 )")
training_data_input = int(input())

if training_data_input == 1:
    training_data = pandas.read_csv('Desktop/training_data_mixed_blade_half1.csv') # NewBlade001, OldBlade001
    training_data_choice = 'training_data_mixed_blade_half1 (NewBlade001, OldBlade001)'

elif training_data_input == 2:
    training_data = pandas.read_csv('Desktop/training_data_mixed_blade_half2.csv') # NewBlade003, OldBlade003 
    training_data_choice = 'training_data_mixed_blade_half2 (NewBlade003, OldBlade003)'

elif training_data_input == 3:
    training_data = pandas.read_csv('Desktop/training_data_mixed_blade.csv') # NewBlade001, NewBlade003, OldBlade001, OldBlade003 
    training_data_choice = 'training_data_mixed_blade (NewBlade001, NewBlade003, OldBlade001, OldBlade003)'

X2 = training_data[['pCut::Motor_Torque',
    'pCut::CTRL_Position_controller::Actual_position',
    'pCut::CTRL_Position_controller::Actual_speed',
    'pSvolFilm::CTRL_Position_controller::Actual_position',
    'pSvolFilm::CTRL_Position_controller::Actual_speed',
    'pCut::CTRL_Position_controller::Lag_error',
    'pSvolFilm::CTRL_Position_controller::Lag_error'
    ]]
Y2 = training_data[['Replaced blade']].values.ravel()

### End ###


### Import testing data ###

print("Which testing dataset would you like to use?")
print("1.   testing_data_new (NewBlade002)")
print("2.   testing_data_worn (OldBlade002)")
print("3.   testing_data_mixed (First 1000 of NewBlade002 and OldBlade002)")
testing_data_input = int(input())

if testing_data_input == 1:
    testing_data = pandas.read_csv('Desktop/testing_data_new.csv') # NewBlade002
    testing_data_choice = 'testing_data_new (NewBlade002)'

elif testing_data_input == 2:
    testing_data = pandas.read_csv('Desktop/testing_data_worn.csv') # OldBlade002
    testing_data_choice = 'testing_data_worn (OldBlade002)'

elif testing_data_input == 3:
    testing_data = pandas.read_csv('Desktop/testing_data_mixed.csv') # First 1000 of NewBlade002 and OldBlade002
    testing_data_choice = 'testing_data_mixed (First 1000 of NewBlade002 and OldBlade002)'

test_data = testing_data[['pCut::Motor_Torque',
    'pCut::CTRL_Position_controller::Actual_position',
    'pCut::CTRL_Position_controller::Actual_speed',
    'pSvolFilm::CTRL_Position_controller::Actual_position',
    'pSvolFilm::CTRL_Position_controller::Actual_speed',
    'pCut::CTRL_Position_controller::Lag_error',
    'pSvolFilm::CTRL_Position_controller::Lag_error'
    ]]

### End ###

### Scale the data (Mode = 0, Standard Deviation = 1) ###

print("Would you like to scale the data? (Mean and Standard Deviation standardisation")
print("1.   Yes")
print("2.   No")
scale_data_input = int(input())

if scale_data_input == 1:
    scaler = StandardScaler()
    X2 = scaler.fit_transform(X2)
    test_data = scaler.transform(test_data)
    scale_data_choice = 'Data scaled'

elif scale_data_input == 2:
    scale_data_choice = 'Data not scaled'

### End ###

### Classes ###

class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        
    def fit(self, X2, Y2):
        trn_start_time = time.time()
        self.rf_model.fit(X2,Y2)
        trn_end_time = time.time()
        self.trn_time = trn_end_time - trn_start_time        

    def prediction(self, test_data):
        pred_start_time = time.time()
        rf_y_prob = self.rf_model.predict_proba(test_data)[:, 1]
        pred_end_time = time.time()
        pred_time = pred_end_time - pred_start_time
        print("Random Forest:", rf_y_prob.mean(), "Training time", self.trn_time, 'Prediction time:', pred_time)
    
class LogisticRegressionModel:
    def __init__(self, max_iter=1000, random_state=42):
        self.log_model = LogisticRegression(max_iter=max_iter, random_state=random_state)

    def fit(self, X2, Y2):
        trn_start_time = time.time()
        self.log_model.fit(X2,Y2)
        trn_end_time = time.time()
        self.trn_time = trn_end_time - trn_start_time

    def prediction(self, test_data):
        pred_start_time = time.time()
        log_y_prob = self.log_model.predict_proba(test_data)[:, 1]
        pred_end_time = time.time()
        pred_time = pred_end_time - pred_start_time
        print("Logistic Regression:", log_y_prob.mean(), "Training time", self.trn_time, 'Prediction time:', pred_time)

class DecisionTreeModel:
    def __init__(self, random_state=42):
        self.dt_model = DecisionTreeClassifier(random_state=random_state)

    def fit(self, X2, Y2):
        trn_start_time = time.time()
        self.dt_model.fit(X2,Y2)
        trn_end_time = time.time()
        self.trn_time = trn_end_time - trn_start_time

    def prediction(self, test_data):
        pred_start_time = time.time()
        dt_y_prob = self.dt_model.predict_proba(test_data)[:, 1]
        pred_end_time = time.time()
        pred_time = pred_end_time - pred_start_time
        print("Decision Tree:", dt_y_prob.mean(), "Training time", self.trn_time, 'Prediction time:', pred_time)

class SupportVectorClassificationModel:
    def __init__(self, probability=True, random_state=42):
        self.svc_model = SVC(probability=probability, random_state=random_state)

    def fit(self, X2, Y2):
        trn_start_time = time.time()
        self.svc_model.fit(X2,Y2)
        trn_end_time = time.time()
        self.trn_time = trn_end_time - trn_start_time

    def prediction(self, test_data):
        pred_start_time = time.time()
        svc_y_prob = self.svc_model.predict_proba(test_data)[:, 1]
        pred_end_time = time.time()
        pred_time = pred_end_time - pred_start_time
        print("Support Vector Classification:", svc_y_prob.mean(), "Training time", self.trn_time, 'Prediction time:', pred_time)
    
class KNearestNeighboursModel:
    def __init__(self, n_neighbors=5):
        self.knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X2, Y2):
        trn_start_time = time.time()
        self.knn_model.fit(X2,Y2)
        trn_end_time = time.time()
        self.trn_time = trn_end_time - trn_start_time

    def prediction(self, test_data):
        pred_start_time = time.time()
        knn_y_prob = self.knn_model.predict_proba(test_data)[:, 1]
        pred_end_time = time.time()
        pred_time = pred_end_time - pred_start_time
        print("K Nearest Neighbour:", knn_y_prob.mean(), "Training time", self.trn_time, 'Prediction time:', pred_time)
           
class GradientBoostingClassifierModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.gbc_model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X2, Y2):
        trn_start_time = time.time()
        self.gbc_model.fit(X2,Y2)
        trn_end_time = time.time()
        self.trn_time = trn_end_time - trn_start_time

    def prediction(self, test_data):
        pred_start_time = time.time()
        gbc_y_prob = self.gbc_model.predict_proba(test_data)[:, 1]
        pred_end_time = time.time()
        pred_time = pred_end_time - pred_start_time
        print("Gradient Boosting Classifier:", gbc_y_prob.mean(), "Training time", self.trn_time, 'Prediction time:', pred_time)

class AdaBoostModel:
    def __init__(self, n_estimators=100, random_state=42, algorithm = 'SAMME'):
        self.abc_model = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state, algorithm=algorithm)

    def fit(self, X2, Y2):
        trn_start_time = time.time()
        self.abc_model.fit(X2,Y2)
        trn_end_time = time.time()
        self.trn_time = trn_end_time - trn_start_time

    def prediction(self, test_data):
        pred_start_time = time.time()
        abc_y_prob = self.abc_model.predict_proba(test_data)[:, 1]
        pred_end_time = time.time()
        pred_time = pred_end_time - pred_start_time
        print("Ada Boost", abc_y_prob.mean(), "Training time", self.trn_time, 'Prediction time:', pred_time)

### End ###

### START ###

menu_opt = -1

while menu_opt != 0:
    print("Welcome to the Predictive Maintenance program. Please state which function you would like to use:")
    print("   1. Random Forest.")
    print("   2. Logistic Regression.")
    print("   3. Decision Trees.")
    print("   4. Support Vector Classification.")
    print("   5. K Nearest Neighbours")
    print("   6. Gradient Boosting Machines")
    print("   7. AdaBoost")
    print("   8. All results")
    print("   0. Exit.")
    menu_opt = int(input())
    
    if menu_opt == 1:
        print("Random Forest")
        rf_model = RandomForestModel()
        rf_model.fit(X2, Y2)
        print(rf_model.prediction(test_data))

    elif menu_opt == 2:
        print("Logistic Regression")
        log_model = LogisticRegressionModel()
        log_model.fit(X2, Y2)
        print(log_model.prediction(test_data))
        
    elif menu_opt == 3:
        print("Decision Tree")
        dt_model = DecisionTreeModel()
        dt_model.fit(X2, Y2)
        print(dt_model.prediction(test_data))

    elif menu_opt == 4:
        print("Support Vector Classification")
        svc_model = SupportVectorClassificationModel()
        svc_model.fit(X2, Y2)
        print(svc_model.prediction(test_data))
    
    elif menu_opt == 5:
        print("K Nearest Neighbours")
        knn_model = KNearestNeighboursModel()
        knn_model.fit(X2, Y2)
        print(knn_model.prediction(test_data))

    elif menu_opt == 6:
        print("Gradient Boosting Machines")
        gbc_model = GradientBoostingClassifierModel()
        gbc_model.fit(X2, Y2)
        print(gbc_model.prediction(test_data))
        
    elif menu_opt == 7:
        print("AdaBoost")        
        abc_model = AdaBoostModel()
        abc_model.fit(X2, Y2)
        print(abc_model.prediction(test_data))

    elif menu_opt == 8:
        print()
        print("All results:")
        print()
        print("Training data set:", training_data_choice)
        print("Testing data set:", testing_data_choice)
        print(scale_data_choice)
        print()
        rf_model = RandomForestModel()
        rf_model.fit(X2, Y2)
        rf_model.prediction(test_data)

        log_model = LogisticRegressionModel()
        log_model.fit(X2, Y2)
        log_model.prediction(test_data)

        dt_model = DecisionTreeModel()
        dt_model.fit(X2, Y2)
        dt_model.prediction(test_data)

        svc_model = SupportVectorClassificationModel()
        svc_model.fit(X2, Y2)
        svc_model.prediction(test_data)

        knn_model = KNearestNeighboursModel()
        knn_model.fit(X2, Y2)
        knn_model.prediction(test_data)

        gbc_model = GradientBoostingClassifierModel()
        gbc_model.fit(X2, Y2)
        gbc_model.prediction(test_data)

        abc_model = AdaBoostModel()
        abc_model.fit(X2, Y2)
        abc_model.prediction(test_data)

        input()

    elif menu_opt == 0:
        print("Thank you. Goodbye.")
        break


