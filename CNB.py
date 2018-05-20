import math
import numpy as np
from scipy import spatial
from classes import *
import statsmodels.api as sm
import lifelines
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score 


class CNB(Predictor):
    '''
    Censoring based Naive Bayes method for survial prediction 
    '''
    def __init__(self, dataMatrix):
        self._N = len(dataMatrix)
        self._nf = len(dataMatrix[1,3:])
        self._dataMatrix = dataMatrix
        self._sBasic = SurvivalBasic(dataMatrix)
        self._Nevent = self._sBasic.get_event_N()
        self._eT = self._sBasic.get_event_ts()
        self._uniqT = self._sBasic.get_unique_T()
        self._S_mu_g = None
        self._S_mu_l = None
        self._S_sigma2_g = None
        self._S_sigma2_l = None
        self._trainFMean = np.zeros(self._nf)
        ## Checking point


    def train(self, trainMatrix):

        self._trainFMean = trainMatrix[:,3:].mean(0)
        train_matrix = np.concatenate((trainMatrix[:,:3] , trainMatrix[:,3:] - self._trainFMean), axis=1)

        n_uniqT = len(self._uniqT)
        unique_etn = len(self._eT)
        
        eventMatrix = np.zeros((self._Nevent,self._nf))
        eventMatrix = train_matrix[train_matrix[:,1] == 1]

        ## Get the index among caes that where the subject should be inserted among the t list
        tE_index = np.zeros(n_uniqT)
        n_pTE = np.zeros(n_uniqT)
        j_t = 0
        for i_t in range(len(self._uniqT)):
            while self._uniqT[i_t] > self._eT[j_t] and j_t < unique_etn-1:
                j_t = j_t + 1
            tE_index[i_t] = j_t 
                # every thing before (<) j_t
            n_pTE[i_t] = j_t
                # j_t individuals before (<) j_t

        ### Step 1: Estimates the Normal parametes for features
        mu_g = np.zeros((n_uniqT,self._nf))
        mu_l = np.zeros((n_uniqT,self._nf))
        sigma2_g = np.zeros((n_uniqT,self._nf))
        sigma2_l = np.zeros((n_uniqT,self._nf))
        ## Weight for parameters
        w_mu_g = np.zeros((n_uniqT,self._nf))
        w_mu_l = np.zeros((n_uniqT,self._nf))
        w_sigma2_g = np.zeros((n_uniqT,self._nf))
        w_sigma2_l = np.zeros((n_uniqT,self._nf))


        for i_t in range(n_uniqT):
            t = self._uniqT[i_t]
            pos_t = list(train_matrix[:,2]).index(t)
            ## Estimation for mu_g and sigma2_g
            if pos_t == self._N-1:
                mu_g[i_t] = 0
                sigma2_g[i_t] = 1
            else:
                mu_g[i_t] = train_matrix[pos_t:, 3:].mean(0)
                sigma2_g[i_t] = train_matrix[pos_t:, 3:].var(0)
                for loop in range(self._nf):
                    if sigma2_g[i_t, loop] == 0.0:
                        sigma2_g[i_t, loop] = 1
            n_g = len(train_matrix[pos_t:])

            w_mu_g[i_t] = max(n_g, 1)/(sigma2_g[i_t].astype(float))
            w_sigma2_g[i_t] = max(n_g-1, 1)/(2*((sigma2_g[i_t]**2).astype(float)))

            ## Estimation for mu_l and sigma2_l

            if tE_index[i_t] == 0:
                mu_l[i_t] = 0
                sigma2_l[i_t] = 1
            else:
                mu_l[i_t] = eventMatrix[:tE_index[i_t], 3:].mean(0)
                sigma2_l[i_t] = eventMatrix[:tE_index[i_t], 3:].var(0)
                for loop in range(self._nf):
                    if sigma2_l[i_t, loop] == 0.0:
                        sigma2_l[i_t, loop] = 1
            n_e_l = len(eventMatrix[:tE_index[i_t]])

            w_mu_l[i_t] = max(n_e_l, 1)/(sigma2_l[i_t].astype(float))
            w_sigma2_l[i_t] = max(n_e_l-1, 1)/(2*((sigma2_l[i_t]**2).astype(float)))
 

        mu_g = w_mu_g*mu_g
        mu_l = w_mu_l*mu_l
        sigma2_g = w_sigma2_g*sigma2_g
        sigma2_l = w_sigma2_l*sigma2_l

        ## Apply smoothing parameters
        lowess = sm.nonparametric.lowess
        # Note a transpose between mu/sigma2_* and S_mu/sigma2_*
        self._S_mu_g = np.zeros((self._nf, n_uniqT))
        self._S_mu_l = np.zeros((self._nf, n_uniqT))
        self._S_sigma2_g = np.zeros((self._nf, n_uniqT))
        self._S_sigma2_l = np.zeros((self._nf, n_uniqT))
        

        for i_f in range(self._nf):
            print i_f,
            self._S_mu_g[i_f] = np.asarray(lowess(list(mu_g[:, i_f]), list(self._uniqT))[:,1].T)
            self._S_mu_l[i_f] = np.asarray(lowess(list(mu_l[:, i_f]), list(self._uniqT))[:,1].T)
            self._S_sigma2_g[i_f] = np.asarray(lowess(list(sigma2_g[:, i_f]), list(self._uniqT))[:,1].T)
            self._S_sigma2_l[i_f] = np.asarray(lowess(list(sigma2_l[:, i_f]), list(self._uniqT))[:,1].T)
      
        for i_test_T in range(n_uniqT):
            for i_test_f in range(self._nf):
                if self._S_sigma2_l[i_test_f,i_test_T] < 1.0:
                    self._S_sigma2_l[i_test_f,i_test_T] = 1.0



    def find_parameter(self, para_array, t):
        i_t = 0
        while self._uniqT[i_t] < t and i_t < len(self._uniqT):
            i_t = i_t+1
        t_index = i_t - 1

        if t_index == -1 or t_index == 0:
            return para_array[0]
        elif t > self._uniqT[len(self._uniqT)-1]:
            return para_array[len(self._uniqT)-1]
        else:
            return (para_array[t_index-1] + ((t-self._uniqT[t_index-1])/\
                        (self._uniqT[t_index]-self._uniqT[t_index-1])*\
                        (para_array[t_index] - para_array[t_index-1])))


    def predict(self, testMatrix, t):
        N_test = len(testMatrix)
        self._risk = np.zeros(N_test)
        km_t = self._sBasic.sf_KM(t)

        # Centering all features using training mean
        test_Matrix = testMatrix - self._trainFMean

        numerator = 0.0
        denominator_1 = 0.0
        denominator_2 = 0.0
        for i_test in range(N_test):
            for i_f in range(self._nf):
                mu_g = self.find_parameter(self._S_mu_g[i_f], t)
                mu_l = self.find_parameter(self._S_mu_l[i_f], t)
                s2_g = self.find_parameter(self._S_sigma2_g[i_f], t)
                s2_l = self.find_parameter(self._S_sigma2_l[i_f], t)

                ## Calculate the summantion on log scale
                ln_gt = (-0.5*math.log(s2_g) - math.log(((test_Matrix[i_test,i_f] - mu_g)**2)/(2*s2_g)) )
                ln_lt = (-0.5*math.log(s2_l) - math.log(((test_Matrix[i_test,i_f] - mu_l)**2)/(2*s2_l)) )
                numerator = numerator + ln_gt
                denominator_1 = denominator_1 + ln_gt
                denominator_2 = denominator_2 + ln_lt
            if  math.exp(denominator_1) + math.exp(denominator_2) <= 0:
                log_S = math.log(km_t) + numerator - ( max(denominator_1+ math.log(km_t), denominator_2+ math.log(1-km_t)) )
            else:
                log_S = math.log(km_t) + numerator - math.log(math.exp(denominator_1+ math.log(km_t)) + math.exp(denominator_2+ math.log(1-km_t)))
            self._risk[i_test] = 1 - math.exp(log_S)
        # print self._risk[:50]
        return self._risk


    def accuracy(self, true_event, cutoff):
        trueEvent = (true_event)
        ## Make prediction based on the cut-off value
        self._predict_event = (self._risk >= cutoff)*1
        #print list(trueEvent).count(0)
        #print list(self._predict_event).count(0)
        #print (trueEvent - self._predict_event)
        accuracy = np.mean(((trueEvent - self._predict_event) == 0)*1.0)
        # print("Accuracy is %.2f." %(accuracy))  
        return accuracy

    def roc(self, true_event):
        y_true = list(true_event)
        y_scores = list(self._risk)
        roc = roc_auc_score(y_true, y_scores)
        print roc



