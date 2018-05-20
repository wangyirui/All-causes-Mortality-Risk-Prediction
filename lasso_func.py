import os
import argparse
import sys
import pickle
import math
import numpy as np
import csv
from scipy import spatial
from classes import *
from cvxopt import matrix
from cvxopt import solvers
from numpy.linalg import matrix_rank
'''
HOW TO CALL:
beta = train(sortedTrain[:,0], sortedTrain[:,1], sortedTrain[:,2], sortedTrain[:,3:])
'''
class lasso():
    def __init__(self, sdataMatrix):
        self.beta = []
        KM = ll.KaplanMeierFitter()
        kmf = KM.fit(sdataMatrix[:,2], event_observed=sdataMatrix[:,1]).survival_function_
        self._kmf = np.zeros((np.shape(kmf)[0] ,2))
        self._kmf[:,0] = np.asarray(list(kmf.index))
        self._kmf[:,1] = list(np.asarray(kmf))

    def construct_R(self, id_list, death_list, end_time_list, featureMatrix):

        # Get the order based on retire time
        retire_order = np.argsort(end_time_list)

        # Sort data
        sorted_id = np.array(id_list)[retire_order]
        sorted_death = np.array(death_list)[retire_order]
        sorted_end_time = np.array(end_time_list)[retire_order]
        sorted_featureMatrix = featureMatrix[retire_order, :]

        # Find true death event (ignore retirement)
        death_time = set()
        all_death = []
        instance_len = len(sorted_id)
        for i in range(instance_len):
            if sorted_death[i] == 1:
                death_time.add(sorted_end_time[i])
                all_death.append(sorted_end_time[i])
        death_time = list(death_time)
        death_time = sorted(death_time)

        # Construct R sets
        R_list = []
        for time in death_time:
            group = []
            for index in range(instance_len):
                if sorted_end_time[index] >= time:
                    group.append(sorted_id[index])
            R_list.append(group)

        return R_list, death_time, all_death

    def accuracy(self, true_event, cutoff):
		trueEvent = np.matrix(true_event).T
		self._predict_event = (self._risk >= cutoff)*1
		accuracy = np.mean(((trueEvent - self._predict_event) == 0)*1.0)
		print("Accuracy is %.2f." %(accuracy))  
		return accuracy

    def predict(self, test_Matrix, t):
        xbeta = np.exp((np.dot(test_Matrix,self.beta)))
        
        bl_sur = 1.0
        for i_t in range(len(self._kmf)):
            if self._kmf[i_t, 0] > t:
                bl_sur = self._kmf[i_t-1, 1]
                break
        sur = np.power(bl_sur, xbeta)
        self._risk = 1 - sur

        print self._risk

        return self._risk

    def train(self, id_list, death_list, end_time_list, featureMatrix):
        
        # Construct R sets
        R_list,death_time, all_death = self.construct_R(id_list, death_list, end_time_list, featureMatrix)
        
        print "R set constructed!"
        ## Solve lasso cox likelihood

        #  STEP 1: Fix s and initialize beta_hat = 0
        beta_new = np.zeros(int(np.shape(featureMatrix)[1]))
        beta_record = np.zeros(int(np.shape(featureMatrix)[1]))
        delta = np.zeros(len(id_list))
        eta = np.zeros(len(id_list))
        u = np.zeros(len(id_list))
        A = np.zeros(len(id_list))
        z = np.zeros(len(id_list))

        # Main loop
        max_iter = 10
        s = 11.97 * 0.44
        for iter in range(max_iter):   
            print iter       
            for i in range(len(id_list)):
                person_id = i + 1
                delta[i] = 1 if death_list[i] == 1 else 0

                # STEP 2: Compute eta, u, A and z based on the current value of beta_hat
                eta[i] = np.dot(np.array(featureMatrix[i,:]), beta_new)

                sum1 = 0.0
                sum2 = 0.0

                for j in range(len(R_list)):
                    if person_id in R_list[j]:
                        d_k = all_death.count(death_time[j])
                        r_list_int = [int(x) for x in R_list[j]]
                        denominator = np.sum(np.exp(eta[np.array(np.array(r_list_int) - 1)]))
                        sum1 += d_k / denominator
                        sum2 += d_k * (np.exp(eta[i]) / denominator)**2

                u[i] = delta[i] - np.exp(eta[i]) * sum1
                A[i] = -np.exp(eta[i]) * sum1 +  sum2
                A[i] = -1.0 * A[i]
                z[i] = eta[i] + u[i] / A[i]
            
                
            # SETP 3: Minimize (z-Xbeta)'A(z-Xbeta) subject to Sigma|beta_i| <= s

            # SUB-STEP 1: Start with E={i_0} where delta_i0 = sign(beta^hat_0), beta^hat_0 beingthe overall least squared estimate
            if iter == 0:
                # Unconstraint lasso
                z_mat = np.matrix(z).T
                X_mat = np.matrix(featureMatrix)
                beta_mat = np.matrix(beta_new).T
                eta_mat = np.matrix(eta).T
                A_mat = np.identity(len(id_list)) * A
                learning_rate = 1.0
                beta_old = beta_mat
                beta_new = beta_old

                gradient_old = np.zeros(int(np.shape(featureMatrix)[1]))
                for loop in range(10):
                    gradient = (-2.0 * z_mat.T * A_mat * X_mat).T + (2.0 * X_mat.T * A_mat * X_mat * beta_old)
                    gradient_old = gradient
                    beta_new = beta_old + learning_rate * gradient  

                    if np.sum(np.abs(beta_new - beta_old)) < 0.1:
                        break               
                    beta_old = beta_new
                # Compute a sign vector
                sign = []
                beta_new = np.array(beta_new)
                for index in range(len(beta_new)):
                    if beta_new[index] >= 0:
                        sign.append(1.0)
                    else:
                        sign.append(-1.0)
                sign = np.matrix(sign)
                initialized = 0

                #np.insert(sign_matrix, int(np.shape(sign_matrix)[0]), values = sign, axis = 0)

            # SUB-STEP 2 3 4: Find beta^hat to minimize g(beta) subject to GeBeta <= t*1

            while(1): 
                if initialized == 0:
                    # Create a sign matrix
                    sign_matrix = sign
                    initialized = 1
                else:
                    # Compute a sign vector
                    sign = []
                    beta_new = np.array(beta_new)
                    for index in range(len(beta_new)):
                        if beta_new[index] >= 0:
                            sign.append(1.0)
                        else:
                            sign.append(-1.0)
                    sign = np.matrix(sign)
                    sign_matrix = np.insert(sign_matrix, int(np.shape(sign_matrix)[0]), values = sign, axis = 0)

                P_matrix = np.matrix(featureMatrix).T * np.diag(A) * np.matrix(featureMatrix)
                P = matrix(P_matrix, tc = 'd')

                
                q_matrix = - 2.0 * np.matrix(z) * np.diag(A) * np.matrix(featureMatrix)
                q = matrix(q_matrix.T, tc = 'd')
                #q = matrix(np.zeros(int(np.shape(featureMatrix)[1])), tc = 'd')

                G = matrix(sign_matrix, tc = 'd')

                h = matrix(s * np.ones(np.shape(G)[0]), tc = 'd')

                sol = solvers.qp(P,q,G,h)
                solved_beta = np.array(sol['x'])

                for pos in range(len(solved_beta)):
                    if np.abs(solved_beta[pos]) <  1e-7:
                        solved_beta[pos] = 0.0
                beta_new = solved_beta
                
                print np.sum(np.abs(beta_new))
                if np.sum(np.abs(beta_new)) <= s:
                    break

            print np.sum(np.abs(beta_record - beta_new)) / int(np.shape(featureMatrix)[1])
            if np.sum(np.abs(beta_record - beta_new)) < 0.01 * int(np.shape(featureMatrix)[1]):
                break
            else:
                beta_record = beta_new

        print "final beta:"
        print beta_new
        self.beta = beta_new
            
