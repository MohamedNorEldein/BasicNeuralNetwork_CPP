
import pandas as pd
import numpy as np
import NeuralNetworksLib
import matplotlib.pyplot as plt

def normalize( ar1 :list):
    for i in range(0,len(ar1)):
        ar1[i] = ar1[i]/255
    return ar1


def normalizeList( ar1):
    for i in range(0,len(ar1)):
        normalize(ar1[i])

def listRound(ar1:list):
    for i in range(0,len(ar1)):
            for j in range(0,len(ar1[i])):
                ar1[i][j] = round(ar1[i][j],3)
    return ar1    

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
    y = dataF["label"]
    normalizeList(x)
    
    return x,y


def readTestData(string:str):
    dataF = pd.read_csv(string)
    dataF.dropna()

    x = dataF.iloc[:,:].values.astype(float)
    
    return x


def main():

    train_x , _y = readTrainData("digit-recognizer\\train.csv")
   # test_x  = readTestData("digit-recognizer\\test.csv")
    a = len(train_x) * 3 // 4
    a=1000
    train_y = pd.get_dummies(_y.values).values.astype(float)
    nn = NeuralNetworksLib.NeuralNetwork(train_x.shape[1])
    nn.addLayer(50,activation="relu")
    nn.addLayer(train_y.shape[1],activation="sigmoid")

    print("start reading weight matrix \n" )
    

    #nn.fromfile("./file.csv")
    nn.getWeight(0).Generate(10000)
    nn.getWeight(1).Generate(10000)

    print("finish reading\n" )
    
    nn.train( train_x[:a],train_y[:a],1000,0.01)
    
    print(nn.calcOutput(train_x[0]),"\n", train_y[0])
    #nn.tofile("file.csv")
    
    print(f"end \n ")
    test_x = train_x[a:]
    test_y = _y[a:]

   # for i in range(0,30):
   #     print(i , dummies_to_numbers(nn.calcOutput(test_x[i])),test_y[i])
        #plt.imshow(test_x[i].reshape(28,28),cmap='gray')
        #plt.show()


if(__name__=='__main__'):
    
    dataF = pd.read_csv("digit-recognizer/train.csv")
    dataF.dropna()

    main()
    input("end\n")
