
import pandas as pd
import numpy as np
import NeuralNetworksLib



def main():
    data = pd.read_csv("nba_salary_stats.csv")
    data.dropna()

    tf1 = NeuralNetworksLib.CreateTensorFromPandasDataFrame(data,["fg","fga"])
    tf2 = NeuralNetworksLib.Tensor(len(data.index),1,data["salary"].array.tolist())    

    tf1.print()
    tf2.print()

    tf3 = tf1.getCorelationMatrix()*tf2 * 2 
    tf3.print()

def normalize( ar1 ):
    mx = ar1.max()
    mn = ar1.min()
    ar1 = (ar1 - mn)/(mx-mn)
    return ar1


if(__name__=='__main__'):
    
   
    data = pd.read_csv("nba_salary_stats.csv")
    data.dropna()


    nn = NeuralNetworksLib.NeuralNetwork(1)
    nn.addLayer(1)

    ar1 = data["fg"]
    ar2 = data["fga"]

    ar1 = normalize(ar1)
    ar2 = normalize(ar2)
    
    nn.train(ar1,ar2,100,0.01)    
    