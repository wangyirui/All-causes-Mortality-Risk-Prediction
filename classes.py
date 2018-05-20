
from abc import ABCMeta, abstractmethod
import numpy as np
import lifelines as ll


class Instance:
    def __init__(self, id, death, end_time, feature_vector):
        self._id = id
        self._death = death
        self._end_time = end_time
        self._feature_vector = feature_vector



class SurvivalBasic:

    def __init__(self, dataMatrix):
        ## We will fix the first three features in order: id, event, time
        self._sortedMatrix = dataMatrix[dataMatrix[:,2].argsort()]
        self._N = len(dataMatrix)
        # Sorted time list
        self._sorted_t = list(np.argsort(dataMatrix[:,2]))
        # Total number of events
        self._Nevent = np.count_nonzero(dataMatrix[:,1])
		# Construct unique event time list
        self._event_ts = set()
        t_list = dataMatrix[dataMatrix[:,1] == 1,2]
        for t in t_list :
        	self._event_ts.add(t)
        self._event_ts = sorted(self._event_ts)
        # print self._event_ts
        # Number of distinct event time
        self._n_et = len(self._event_ts)
        sorted_t = self._sortedMatrix[:,2]
        self._uniqT = set()
        for t in sorted_t:
            self._uniqT.add(t)
        self._uniqT = sorted(self._uniqT)
        self._n_uniqT = len(self._uniqT)
        # Initial array to store KM survival function
        self._KM_sf = np.zeros(self._n_uniqT)
        self._kmf = None

    def get_unique_T(self):
        return self._uniqT

    def get_sortedMatrix(self):
        return self._sortedMatrix

    def get_event_ts(self):
        return self._event_ts

    def get_event_N(self):
        return self._Nevent

    def sf_KM(self, t_point):

        KM = ll.KaplanMeierFitter()
        kmf = KM.fit(self._sortedMatrix[:,2], event_observed=self._sortedMatrix[:,1]).survival_function_
        self._kmf = np.zeros((np.shape(kmf)[0] ,2))
        self._kmf[:,0] = np.asarray(list(kmf.index))
        self._kmf[:,1] = list(np.asarray(kmf))

        for i_t in range(len(self._kmf)):
            bl_sur = 1.0
            if self._kmf[i_t, 0] > t_point:
                bl_sur = self._kmf[i_t-1, 1]
                break
        return bl_sur

        '''      
        sorted_all_t = self._sortedMatrix[:,2]
        
        # Think about whether need this
        for i_t in range(self._n_uniqT):
            t = self._uniqT[i_t]
            if i_t != self._n_uniqT -1: 
                t_next = self._uniqT[i_t+1]
            pos_t = list(sorted_all_t).index(t)
            if i_t != self._n_uniqT -1: 
                pos_t_next = list(sorted_all_t).index(t_next)
            else:
                pos_t_next = self._N
            
            if self._sortedMatrix[pos_t,1] == 0: 
                p_t = 1
            else:
                p_t = 1- np.count_nonzero(self._sortedMatrix[pos_t:pos_t_next,1])/(self._N-pos_t)
            if i_t == 0:
                self._KM_sf[i_t] = p_t
            else:
                self._KM_sf[i_t] = self._KM_sf[i_t-1]*p_t
        i_t = 0
        while self._uniqT[i_t] < t_point and i_t < len(self._uniqT):
            i_t = i_t+1
        t_stop = i_t-1
        return self._KM_sf[t_stop]
        
        '''




# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, trainMatrix): 
        pass

    @abstractmethod
    def predict(self, testMatrix, t, cutoff):
        pass



