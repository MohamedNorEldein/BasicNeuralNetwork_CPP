
import pandas as pd
import numpy as np
import NeuralNetworksLib
import math
import matplotlib.pyplot as plt


def dummies_to_numbers(dummies_list:list):
    a    =0
    index =0
    for i in range(0,len(dummies_list)):
        if(dummies_list[i][0] > a):
            a=dummies_list[i][0]
            index =i

    return a,index

def readTrainData(string:str):
    dataF = pd.read_csv(string)
    dataF.dropna()

    a:int = len(dataF)

    x = dataF.iloc[:a,1:].values.astype(float)
    y = pd.get_dummies(dataF["label"]).values[:a].astype(float)
    
    return x,y


def readTestData(string:str):
    dataF = pd.read_csv(string)
    dataF.dropna()

    x = dataF.iloc[:,:].values.astype(float)
    
    return x


def main():

    train_x , train_y = readTrainData("digit-recognizer\\train.csv")
    test_x  = readTestData("digit-recognizer\\test.csv")


    nn = NeuralNetworksLib.NeuralNetwork(train_x.shape[1])
    nn.addLogisticLayer(train_y.shape[1])

    print("start reading weight matrix \n" )

    array = pd.read_csv("./file.csv").iloc[:,1:]
    nn.getWeight(0).fromlist(array.values)
    #nn.getWeight(0).print()
    
    print("finish reading\n" )

    #nn.train(train_x,train_y,5,0.8)
    w1 = nn.getWeightsAsDataFrame()
    w1.to_csv("file.csv")
    
    print(f"end \n ")
    
    for i in range(0,30):
        print(i , dummies_to_numbers(nn.calcOutput(test_x[i])))
        plt.imshow(test_x[i].reshape(28,28),cmap='gray')
        plt.show()


if(__name__=='__main__'):
    
    dataF = pd.read_csv("digit-recognizer/train.csv")
    dataF.dropna()

    main()
    
