import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

class BM():
    def __init__(self, eta, n):
        self.W = [1,1,1,1]
        self.n = n
        
        self.eta = eta
        
    def model_p(self,weights,spins):
        num = np.exp( - np.sum([-weights[i]*spins[i]*spins[(i+1)%self.n] for i in range(self.n)]))
        
        return num
        
    
    def gen_model_data(self,N):
        list_spins = []
        
        for i in range(0,N):
            spins_0 = np.random.choice([-1,1], self.n)
            
            Fail_condition = True
            
            while Fail_condition:
                n = np.random.choice([0,1,2,3],1)[0]
                
                spins_1      = np.copy(spins_0)
                spins_1[n]  *= -1
                
                p0 = self.model_p(self.W, list(spins_0))
                p1 = self.model_p(self.W, list(spins_1))
                                
                if np.random.rand() < p1/p0:
                    spins_0 = spins_1
                
                else:
                    Fail_condition = False
            
            spins_0 = list(spins_0)
            list_spins.append(spins_0)
            
        
        self.model_data = list_spins
        
    def probs(self):
        model_data = np.copy(self.model_data)
        temp = [np.array_str(model_data[i,:]).replace('[','').replace(']','').replace('\n','') for i in range(0,len(model_data))]
        
        self.frequency_model = Counter(temp)
        self.p_model         = {key : self.frequency_model[key] / len(model_data) for key in self.frequency_model.keys()}
                        
            
    def expected_val(self,spins,N):
        spins = np.array(spins)
        exp_val = np.sum([spins[:,i]*spins[:,(i+1)%self.n] for i in range(self.n)], axis=1)/N
                
        return exp_val
        
        
    def update(self,exp_val_data, exp_val_model):
        self.W = self.W + self.eta*(exp_val_data - exp_val_model)
        
        
    def KL_div(self, p_m, p_d):
        return np.sum([p_d[i] * np.log(p_d[i]/p_m[i]) for i in range(0,len(p_d))])
        
    
    