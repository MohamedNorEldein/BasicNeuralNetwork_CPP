
from TensorLib import*


class NeuralNetwork:
    def __init__(self, inputNum) -> None:
        func = file.CreateNeuralNetwork
        func.argtypes = [ctypes.c_size_t]
        func.restype = ctypes.c_void_p
        self.pointer = func(inputNum)
        print("constructor self.pointer : ",hex (self.pointer),"\n")

        return

    def __del__(self):
        print("__del__ self.pointer : ",self.pointer,"\n")
        
        func = file.DeleteNeuralNetwork
        func.argtypes = [ctypes.c_void_p]
        func(self.pointer)
        return

    def addLayer(self, count:int, data = None):
        func = file.addLayerNeuralNetwork
        func.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]

        if(data == None):
            func(self.pointer,count, 0)
            return
        
        func(self.pointer,count, (ctypes.c_float * len(data))(*data))
        return
    
    def addLogisticLayerNeuralNetwork(self, count:int, data = None):
        func = file.addLogisticLayerNeuralNetwork

        func.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]

        if(data == None):
            func(self.pointer,count, 0)
            return
        
        func(self.pointer,count, (ctypes.c_float * len(data))(*data))
        return

    def calcOutput(self, x:Tensor)->Tensor:
        func  = file.calcOutputNeuralNetwork
        func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        func.restype = ctypes.c_void_p
        t1 = Tensor()
        t1.pointer = func(self.pointer, x.pointer)
        return t1

    def print(self):
        #print("print self.pointer : ",self.pointer,"\n")
        func = file.printNeuralNetworks
        func.argtypes=[ctypes.c_void_p]
        func(self.pointer)

    def train(self , x:list, y:list, count:int, learningRate:float):
        func  = file.trainNeuralNetwork
        func.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p    ,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_float]
        func.restype = ctypes.c_float

        return func(self.pointer,(ctypes.c_float * len(x))(*x),(ctypes.c_float * len(x))(*y),len(x),count, learningRate)

    def learn(self , train_x:list, train_y:list, test_x:list, test_y:list, count:int, learningRate:float):
        func  = file.trainNeuralNetwork
        func.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p    ,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_float]
        func.restype = ctypes.c_float
        n = len(train_x)
        return func(self.pointer,(ctypes.c_float * n)(*train_x),(ctypes.c_float * n)(*train_y),(ctypes.c_float * n)(*test_x),(ctypes.c_float * n)(*test_y), n,count, learningRate)

