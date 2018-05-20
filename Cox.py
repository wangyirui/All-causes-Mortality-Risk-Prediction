import lifelines as ll
# import statsmodels.duration.hazard_regression as sm 
import numpy as np
import pandas as pd
import math
from classes import *
from lifelines.datasets import load_rossi
from lifelines.utils import k_fold_cross_validation
from sklearn.metrics import roc_auc_score

class Cox(Predictor):
	def __init__(self, sdataMatrix):
		#print data
		KM = ll.KaplanMeierFitter()
		kmf = KM.fit(sdataMatrix[:,2], event_observed=sdataMatrix[:,1]).survival_function_
		self._kmf = np.zeros((np.shape(kmf)[0] ,2))
		self._kmf[:,0] = np.asarray(list(kmf.index))
		self._kmf[:,1] = list(np.asarray(kmf))
		self._predict_event = None
        # self._blsuv = sm.PHReg.baseline_cumulative_hazard()
        # suv = sm.PHReg(endog=sdataMatrix[:,2],exog=sdataMatrix[:,3:], status=sdataMatrix[:,1])
        # (endog, exog, status=None, entry=None, strata=None, offset=None, ties='breslow', missing='drop', **kwargs)[source]
        # self._blsuv = baseline_cumulative_hazard


	def train(self, trainMatrix):
		self._trainFMean = trainMatrix[:,3:].mean(0)
		sur_matrix = np.concatenate((trainMatrix[:,1:3] , trainMatrix[:,3:] - self._trainFMean), axis=1)
		data = pd.DataFrame(sur_matrix)
		self._cf = ll.CoxPHFitter()
		self._cf.fit(data, 1, event_col=0)
		bh = self._cf.baseline_hazard_
		self._bh = np.zeros((np.shape(bh)[0] ,2))
		self._bh[:,0] = np.asarray(list(bh.index))
		self._bh[:,1] = np.cumsum(np.asarray(bh))
		
		#scores = k_fold_cross_validation(self._cf, data, 1, event_col=0, k=10)
		#print scores
		#print np.mean(scores)
		#print np.std(scores)

	def predict(self, testMatrix, t):
		test_Matrix = testMatrix - self._trainFMean
		#print np.shape(testMatrix)
		testdata = pd.DataFrame(test_Matrix)
		beta = np.asarray(self._cf.hazards_)
		xbeta = np.exp((np.dot(test_Matrix,beta.T)))
		
		bl_sur = 1.0
		for i_t in range(len(self._kmf)):
			if self._kmf[i_t, 0] > t:
				bl_sur = self._kmf[i_t-1, 1]
				break
		sur = np.power(bl_sur, xbeta)
		self._risk = 1 - sur
		return self._risk

		# print self._cf.predict_hazard_(testdata)

	def accuracy(self, true_event, cutoff):
		trueEvent = np.matrix(true_event).T
		self._predict_event = (self._risk >= cutoff)*1
		accuracy = np.mean(((trueEvent - self._predict_event) == 0)*1.0)
		# print("Accuracy is %.2f." %(accuracy))  
		return accuracy


	def roc(self, true_event):
		y_true = true_event
		y_scores = [x for x in self._risk[:,0]]
		roc = roc_auc_score(y_true, y_scores)
		print roc

	def somersD(self, xbeta):
		pass






