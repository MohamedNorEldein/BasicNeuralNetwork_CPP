
import pandas as pd
import numpy as np
import NeuralNetworksLib
import math


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
    return ar1,(mx-mn),mn


if(__name__=='__main__'):
    
    data = pd.read_csv("nba_salary_stats.csv")
    data.dropna()


    nn = NeuralNetworksLib.NeuralNetwork(1)
    nn.addLayer(1)

    array1 = data["fga"]
    array2 = data["salary"]

    a:int = int (math.sqrt(len(array1)/2))

    ar1,m1,n1 = normalize(array1)
    ar2,m2,n2 = normalize(array2)
    
    nn.batchTrain(ar1,ar2,10,0.01)
    
    er=0
    for i in range(len(array1)//2, len(array1)):
        y1 = round( nn.calcOutput(NeuralNetworksLib.Tensor(1,1, [ar1[i]])).getitem(0,0),3)
        y2 = round (ar2[i],3)
        er+= abs(y1 - y2) 
        print(f"{i} {array1[i]} : {y1} , {y2}")

    print(f"end \n test error is {round(er,3)} ")
