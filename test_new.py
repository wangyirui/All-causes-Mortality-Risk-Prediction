import os
import argparse
import sys
import pickle
import math
import numpy as np
import csv
from scipy import spatial
## Import implemented method
from classes import *
#from naive_bayes import *
from Cox import *
#from CNB import *
from lasso_func import *


def load_data(filename):
    instances = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        for line in reader:
            if reader.line_num == 1:
                continue
            if reader.line_num > 301:
                break
            id_string = line[0]
            death_string = line[1]
            end_time_string = line[2]

            int_id = -1
            int_death = -1
            int_end_time = -1
            try:
                int_id = int(id_string)
                int_death = int(death_string)
                int_end_time = int(end_time_string)
            except ValueError:
                raise ValueError("Unable to convert to integer.")
           
            feature_vector = np.zeros(len(line[3:]))
            for index in range(len(line[3:])):
                try:
                    value = float(line[index + 3])
                except ValueError:
                    raise ValueError("Unable to convert value to float.")
                feature_vector[index] = value
            instance = Instance(int_id, int_death, int_end_time, feature_vector)    
            instances.append(instance)
    return instances


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")
    parser.add_argument("--train-data", type=str, required=True, help="The data to use for training.")
    parser.add_argument("--test-data", type=str, required=True, help="The data to use for testing.")
    args = parser.parse_args()
    return args


## Modified version
def create_data_matrix(instances):
    # global row_num
    # global col_num
    row_num = len(instances)
    col_num = len(instances[0]._feature_vector)+3
    dataMatrix = np.zeros([row_num, col_num])
    for i in range(row_num):
        dataMatrix[i,0] = int(instances[i]._id)
        dataMatrix[i,1] = int(instances[i]._death)
        dataMatrix[i,2] = instances[i]._end_time
        dataMatrix[i,3:] = instances[i]._feature_vector
    return dataMatrix

def events_befor_t(sortedMatrix, t):
    events_t = np.zeros(len(sortedMatrix))
    for i in range(len(sortedMatrix)):
        events_t[i] = sortedMatrix[i,1]*(sortedMatrix[i,2]<=t)
    return events_t


def eigValPct(eigVals,percentage):
    sortArray = np.sort(eigVals) 
    sortArray = sortArray[-1::-1]
    arraySum = np.sum(sortArray) 
    tempSum = 0
    num = 0
    for lmbda in sortArray:
        tempSum += lmbda
        num += 1
        if tempSum >= arraySum * percentage:
            return num

def pca(dataMatrix):
    # Zero out the mean of the data
    col_num = np.shape(dataMatrix)[1]
    mean_vect = np.mean(dataMatrix, axis = 0)
    zero_mean_mat = dataMatrix - mean_vect

    # Rescale each coordinate
    squared_sigma = np.sum(np.multiply(zero_mean_mat, zero_mean_mat), axis = 1) / col_num
    rescale_mat = zero_mean_mat / (np.mat(np.sqrt(squared_sigma)).T)

    # Compute cov
    cov_mat = np.cov(rescale_mat, rowvar = 0, ddof = 1)
    
    # Compute eigenvalue and eigenvector
    eigVals,eigVects = np.linalg.eig(np.mat(cov_mat))

    # Compute k
    percentage = 0.99
    k = eigValPct(eigVals,percentage)

    # Choose first k eigVals
    eigValIndex = np.argsort(eigVals)  
    eigValIndex = eigValIndex[-1:-(k + 1):-1] 
    redEigVects = eigVects[:,eigValIndex]   
    lowDimMat = zero_mean_mat * redEigVects 
    reconMat = (lowDimMat * redEigVects.T) + mean_vect
    return lowDimMat

def dataprocess(dataMatrix):
    col_num = np.shape(dataMatrix[0,3:])[0]
    row_num = np.shape(dataMatrix[:,0])[0]

    mean_vect = np.mean(dataMatrix[:,3:], axis = 0) 
    for i in range(3,col_num):
        dataMatrix[:,i] = dataMatrix[:,i] - mean_vect[i-3]

    # Rescale each coordinate
    squared_sigma = np.sum(np.multiply(dataMatrix[:,3:col_num], dataMatrix[:,3:col_num]), axis = 0) / row_num
    dataMatrix[:,3:col_num] = dataMatrix[:,3:col_num] / (np.mat(np.sqrt(squared_sigma)))

    return dataMatrix

def main():
    #args = get_args()
    # Load the training data.
    train_instances = load_data("training_data.csv")
    test_instances = load_data("testing_data.csv")
    #train_instances = load_data("training_data_small.csv")
    #test_instances = load_data("training_data_small.csv")
    # data matrix
    train_dataMatrix = create_data_matrix(train_instances)
    test_dataMatrix = create_data_matrix(test_instances)
    # Construct basic survival analysis elements
    sur_basic_train = SurvivalBasic(train_dataMatrix)
    sortedTrain = sur_basic_train.get_sortedMatrix()
    
    
    # Events status by 5 years
    events_5y = events_befor_t(test_dataMatrix, 5*365.25)
    events_10y = events_befor_t(test_dataMatrix, 10*365.25)

    ### Lasso Cox Model
    featureMatrix = dataprocess(sortedTrain)

    ### Lasso Cox
    predictor_lasso = lasso(sortedTrain)
    predictor_lasso.train(sortedTrain[:,0], sortedTrain[:,1],sortedTrain[:,2],sortedTrain[:,3:])
    
    # Prediction for 5 years
    predictor_lasso.predict(sortedTrain[:,3:], 365.25*5) 
    accuracy = np.zeros(9)
    for i in range(9):
        cut = float(i+1)/10
        accuracy[i] = predictor_lasso.accuracy(events_5y, cut)
    print accuracy

    # Prediction for 10 years
    predictor_lasso.predict(sortedTrain[:,3:], 365.25*10) 
    accuracy = np.zeros(9)
    for i in range(9):
        cut = float(i+1)/10
        accuracy[i] = predictor_lasso.accuracy(events_10y, cut)
    print accuracy
    
    ### Cox Model
    predictor_Cox = Cox(sortedTrain)
    predictor_Cox.train(sortedTrain)

    # Prediction for 5 years
    predictor_Cox.predict(test_dataMatrix[:,3:], 365.25*5)
    predictor_Cox.roc(events_5y)
    accuracy = np.zeros(9)
    for i in range(9):
        cut = float(i+1)/10
        accuracy[i] = predictor_Cox.accuracy(events_5y, cut)
    print accuracy
    
    # Prediction for 10 years
    predictor_Cox.predict(test_dataMatrix[:,3:], 365.25*10)
    predictor_Cox.roc(events_10y)
    accuracy = np.zeros(9)
    for i in range(9):
        cut = float(i+1)/10
        accuracy[i] = predictor_Cox.accuracy(events_10y, cut)
    print accuracy



    
    ### PAC + Cox
    reduced_data = pca(sortedTrain[:,3:])
    pca_data = np.concatenate((sortedTrain[:,0:3], reduced_data),axis = 1)
    predictor_PCACox = Cox(pca_data)
    predictor_PCACox.train(pca_data)

    reduced_test = pca(test_dataMatrix[:,3:]) 
    pca_test = np.concatenate((test_dataMatrix[:,0:3], reduced_test),axis = 1)

    # Prediction for 5 years
    predictor_PCACox.predict(pca_test[:,3:], 5*365.25)
    # predictor_PCACox.roc(events_5y)

    accuracy = np.zeros(9)
    for i in range(9):
        cut = float(i+1)/10
        accuracy[i] = predictor_PCACox.accuracy(events_5y, cut)
    print accuracy
    

    # Prediction for 10 years
    predictor_PCACox.predict(pca_test[:,3:], 10*365.25)
    # predictor_PCACox.roc(events_10y)

    accuracy = np.zeros(9)
    for i in range(9):
        cut = float(i+1)/10
        accuracy[i] = predictor_PCACox.accuracy(events_10y, cut)
    print accuracy


    '''
    ### Naive Bayes for Censoring
    predictor_CNB = CNB(sortedTrain)
    predictor_CNB.train(sortedTrain)

    # Prediction for 5 years
    predicts_5y = predictor_CNB.predict(test_dataMatrix[:,3:], 5*365.25)
    # predictor_CNB.accuracy(events_5y, 0.5)
    predictor_CNB.roc(events_5y)

    accuracy = np.zeros(9)
    for i in range(9):
        cut = float(i+1)/10
        accuracy[i] = predictor_CNB.accuracy(events_5y, cut)
    print accuracy


    # Prediction for 10 years
    predicts_5y = predictor_CNB.predict(test_dataMatrix[:,3:], 10*365.25)
    # predictor_CNB.accuracy(events_10y, 0.5)
    predictor_CNB.roc(events_10y)

    accuracy = np.zeros(9)
    for i in range(9):
        cut = float(i+1)/10
        accuracy[i] = predictor_CNB.accuracy(events_10y, cut)
    print accuracy
    '''
    ### Performance evaluation on Testing Data
    

if __name__ == "__main__":
    main()