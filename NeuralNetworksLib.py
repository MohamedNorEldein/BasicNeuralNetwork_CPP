
from TensorLib import*
import pandas as pd
import numpy as np
import math

def normalize( ar1 :list):
    mx = max(ar1)
    mn = min(ar1)
    for i in range(0,len(ar1)):
        ar1[i] = (ar1[i] - mn)/(mx-mn)
    return ar1

def listRound(ar1:list):
    for i in range(0,len(ar1)):
            for j in range(0,len(ar1[i])):
                ar1[i][j] = round(ar1[i][j],3)
    return ar1    

class NeuralNetwork:
    def __init__(self, inputNum) -> None:
        func = file.CreateNeuralNetwork
        func.argtypes = [ctypes.c_size_t]
        func.restype = ctypes.c_void_p
        self.pointer = func(inputNum)
        self.layerNum =0
        #print("constructor self.pointer : ",hex (self.pointer),"\n")

        return

    def __del__(self):
        #print("__del__ self.pointer : ",self.pointer,"\n")
        
        func = file.DeleteNeuralNetwork
        func.argtypes = [ctypes.c_void_p]
        func(self.pointer)
        return

    def addLayer(self, count:int, data = None):
        func = file.addLayerNeuralNetwork
        func.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
        self.layerNum += 1
        if(data == None):
            func(self.pointer,count, 0)
            return
        
        func(self.pointer,count, (ctypes.c_float * len(data))(*data))
        return
    
    def addLogisticLayer(self, count:int, data = None):
        func = file.addLogisticLayerNeuralNetwork
        self.layerNum += 1
        func.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]

        if(data == None):
            func(self.pointer,count, 0)
            return
        
        func(self.pointer,count, (ctypes.c_float * len(data))(*data))
        return

    def calcOutputTensor(self, x:Tensor)->Tensor:
        func  = file.calcOutputNeuralNetwork
        func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        func.restype = ctypes.c_void_p
        t1 = Tensor(pointer = func(self.pointer, x.pointer))
        
        return t1
    
    def calcOutput(self, x:list)->list:
        t1 = Tensor()
        t1.init(numrows= len(x),numColumns= 1,python_array= normalize(x))
        return listRound(self.calcOutputTensor(t1).tolist())

    def print(self):
        #print("print self.pointer : ",self.pointer,"\n")
        func = file.printNeuralNetworks
        func.argtypes=[ctypes.c_void_p]
        func(self.pointer)

    def train_1dList(self , x:list, y:list, count:int,num :int, learningRate:float):
        func  = file.trainNeuralNetwork
        func.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p    ,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_float]
        func.restype = ctypes.c_float

        return func(self.pointer,(ctypes.c_float * len(x))(*x),(ctypes.c_float * len(y))(*y),count,num, learningRate)

    def learn_1dList(self , train_x:list, train_y:list, test_x:list, test_y:list, count:int, num :int, learningRate:float):
        func  = file.trainNeuralNetwork
        func.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p    ,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_float]
        func.restype = ctypes.c_float
        
        return func(self.pointer,(ctypes.c_float * len(train_x))(*train_x),(ctypes.c_float * len(train_y))(*train_y),\
                    (ctypes.c_float * len(test_x))(*test_x),(ctypes.c_float * len(train_y))(*test_y), count,num, learningRate)
#------------------------------------------

    def train(self,train_x, train_y, count,learningRate):
        a =  train_x.shape[0]
        ar1 =  train_x.flatten().astype(float).tolist()
        ar2 = train_y.flatten().astype(float).tolist()
        return  self.train_1dList( normalize(ar1), normalize(ar2),a,count,learningRate)

    def batchTrain(self, train_x , train_y,count:int, learningRate:float,b:float=1.6):
        a = int(math.log2(len(train_x)) / math.log2(b))

        for i in range(0,a ):
            print(f"the iteration {i} ,{b**i} ,data done {round(b**i/len(train_x) *100,1)}% processing {round(i/a *100,1)}% \n")
            self.train(train_x[0:int(b**i)],train_y[0:int(b**i)],count,learningRate)
            print("----------------------------------")   
        return

    def getWeight(self,i)->Tensor:
        func = file.getWeight
        func.argtypes = [ctypes.c_void_p,ctypes.c_size_t]
        func.restype = ctypes.c_void_p
        tf = TensorCover(pointer= func(self.pointer, i))
        return tf
    
    def getWeightsAsDataFrame(self):
        ws = []
        for i in range(0, self.layerNum):
            ws.append(self.getWeight(i).tolist())
            
        return pd.DataFrame( self.getWeight(0).tolist())

