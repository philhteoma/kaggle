# Script for kaggle titanic competition

import pandas as pd
import sklearn as sk
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

homeDir = "/home/ray/"
dataDir = homeDir + "kaggle/data/titanic/"

trainDataPath = dataDir + "train.csv"
testDataPath = dataDir + "test.csv"

trainingData = pd.read_csv(trainDataPath, dtype={"Pclass": "object"})
# Columns are: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

# Remove columns I can't work with
trainingData = trainingData.drop(["Name", "Ticket", "Cabin"], axis=1)

###########################
# Set up one hot encoding #
###########################
oneHotFeatures = ["Pclass", "Sex", "Embarked"]

prepDf = trainingData[oneHotFeatures].copy() \
                                     .fillna(value="null")

lc = LabelEncoder()
labelled = prepDf.apply(lc.fit_transform)

ohe = OneHotEncoder()
ohe.fit(labelled)

oneHotTable = ohe.transform(labelled).toarray()

###############################################
# Join One Hot Columns back to starting table #
###############################################

pdOneHot = pd.DataFrame(data=oneHotTable) \
             .reset_index()


nonHotColumns = [x for x in trainingData.columns if x not in oneHotFeatures]
nonHotTable = trainingData[nonHotColumns].copy() \
                          .reset_index()

joinedTable = nonHotTable.join(pdOneHot, ["index"], lsuffix="_l", rsuffix="_r") \
                         .drop("index_r", axis=1) \
                         .drop("index_l", axis=1) \
                         .fillna(value=0)

##############################################
# Split data into training and test sections #
##############################################

joinedTable['is_train'] = np.random.uniform(0, 1, len(joinedTable)) <= .75

trainData, testData = joinedTable[joinedTable['is_train']==True], joinedTable[joinedTable['is_train']==False]

####################
# Train Classifier #
####################

forestClassifier = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

features = [x for x in joinedTable.columns if x not in ["Survived", "is_train"]]

forestClassifier.fit(trainData[features], trainData["Survived"])

###################
# Test Classifier #
###################

result = forestClassifier.predict(testData[features])

preds = pd.DataFrame(data=result)
actual=testData["Survived"]

resultList = result.tolist()
actualList = actual.tolist()

combo = list(zip(resultList, actualList))
review = [True if x[0] == x[1] else False for x in combo]

correct = len([x for x in review if x == True])
incorrect = len([x for x in review if x == False])


print("Total: ", len(review))
print("Correct: ", correct)
print("Incorrect: ", incorrect)
print("Accuracy: {0:.2f}%".format(correct/len(review) * 100))
