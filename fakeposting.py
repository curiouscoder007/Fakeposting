import pickle
import numpy as np
import pandas as pd

class fakeposting():
    def __init__(self,input_data):
        self.input = input_data.reshape(1,6)

#data = pd.DataFrame([[1,1,0,1,1,0]])
#input_data = np.array(data.iloc[0])
#input_data = input_data.reshape(1,6)
    def predict(self):
        filename = 'rfc_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.predict(self.input)
        targets = ['Fake','Real']
        print(result)
        return targets[result[0]]

#print(input_data)
#print('model loaded')
#print(result)