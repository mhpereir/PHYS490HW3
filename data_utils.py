import numpy as np
from collections import Counter

class Data():
    def __init__(self, input_path):
        
        self.input_str    = np.genfromtxt(input_path, dtype=str)
        
        input_spins       = np.array([int(j_str + '1') for i_str in self.input_str for j_str in i_str])
        self.input_data   = input_spins.reshape(-1,4)
        
        temp = [np.array_str(self.input_data[i,:]).replace('[','').replace(']','').replace('\n','') for i in range(0,len(self.input_data))]
        
        
        self.frequency_in = Counter(temp)
        self.p_in         = {key : self.frequency_in[key] / len(self.input_str) for key in self.frequency_in.keys()}