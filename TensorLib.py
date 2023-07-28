"""
    TENSOR CLASS
    is C liberary for linear Algebra with python interface. 

    this script is an application of ctypes liberary in python where the linear algebra class is created in 
    cpp in the file vec.h, a wraper is build over it in C in the file stat.h and stat.cpp
    here a python class in the scipt file to wrape up the C api. 
"""

import ctypes

#the glopal cpp dll

file = ctypes.WinDLL("./build/Release/MyProject.dll")

class Tensor:
    def __init__(self, pointer=0) -> None:

        if(pointer!=0):
            self.pointer = ctypes.c_void_p(pointer)
            return
        self.pointer = 0
        return            

    def init(self,numrows,numColumns, python_array=[]):
        if(len(python_array)==0):
            func = file.createTensorFloat
            func.argtypes = [ctypes.c_size_t,ctypes.c_size_t]
            func.restype = ctypes.c_void_p
            self.pointer = func(numrows,numColumns)
            return
        func = file.initTensorFloat
        func.argtypes = [ctypes.c_size_t,ctypes.c_size_t,ctypes.POINTER(ctypes.c_float)]
        func.restype = ctypes.c_void_p
        c_array = (ctypes.c_float * len(python_array))(*python_array)
        self.pointer = func(numrows,numColumns,c_array)
        return
    
    def getPointer(self):
        return self.pointer
    
    def print(self):
        printf = file.printTensorFloat
        printf.argtypes = [ctypes.c_void_p]
        printf(self.pointer)
        return

    def __del__(self):
        func = file.deleteTensorFloat
        func.argtypes = [ctypes.c_void_p]
        func(self.pointer)

    def __add__(self, other ):
        func = file.addTensorFloat
        func.argtypes = [ctypes.c_void_p,ctypes.c_void_p]
        func.restype = ctypes.c_void_p
        return Tensor(func(self.pointer, other.getPointer()))
         
    def __sub__(self, other ):
        func = file.subTensorFloat
        func.argtypes = [ctypes.c_void_p,ctypes.c_void_p]
        func.restype = ctypes.c_void_p
        return Tensor(func(self.pointer, other.getPointer()))

    
    def __mul__(self, other):
        if(type(other)==Tensor):
            func = file.mullTensorFloat
            func.argtypes = [ctypes.c_void_p,ctypes.c_void_p]
            func.restype = ctypes.c_void_p
            return Tensor(func(self.pointer, other.getPointer()))

        else:
            func = file.scalarmullTensorFloat
            func.argtypes = [ctypes.c_void_p,ctypes.c_float]
            func.restype = ctypes.c_void_p
            return Tensor(func(self.pointer, other.getPointer()))

    def transpose(self):
        func = file.transposeTensorFloat
        func.argtypes = [ctypes.c_void_p]
        func.restype = ctypes.c_void_p
        tf = Tensor()
        tf.pointer = func(self.pointer)
        return tf
    
    def inverse(self):
        func = file.inverseTensorFloat
        func.argtypes = [ctypes.c_void_p]
        func.restype = ctypes.c_void_p
        tf = Tensor()
        tf.pointer = func(self.pointer)
        return tf

    def setitem(self, i, j, value:float):
        func = file.setElementTensorFloat
        func.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_float]
        func(self.pointer,i,j,value)
        return 

    def getitem(self, i, j) -> float:
        func = file.getElementTensorFloat
        func.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t]
        func.restype = ctypes.c_float
        return func(self.pointer,i,j)
         
    def getCorelationMatrix(self):
        func = file.getCorelationMatrix
        func.argtypes = [ctypes.c_void_p]
        func.restype = ctypes.c_void_p
       
        tf = Tensor()
        tf.pointer = func(self.pointer)
        return tf

    def getColumnNum(self) ->int:
        func = file.getColumnNum
        func.argtypes = [ctypes.c_void_p]
        func.restype = ctypes.c_size_t
        return func(self.pointer)
    
    def getRowsNum(self) ->int:
        func = file.getRowsNum
        func.argtypes = [ctypes.c_void_p]
        func.restype = ctypes.c_size_t
        return func(self.pointer)

    def tolist(self)->list:
        n1 = self.getRowsNum()
        n2 = self.getColumnNum()
        out = []
        for i in range(0,n1):
            a = []
            for j in range(0,n2):
                a.append(self.getitem(i,j))
            out.append(a)    
        return out    

    
    def fromlist(self, array):
        n1 = self.getRowsNum()
        n2 = self.getColumnNum()
       
        if(len(array) == n1 and len(array[0])==n2):

            for i in range(0,n1):
                for j in range(0,n2):
                    self.setitem(i,j,array[i][j])
                    
            return   
        else :
            print("dimminsion error")


    
class TensorCover(Tensor):
    def __del__(self):
        return 
    

    


def CreateTensorFromPandasDataFrame(data, KeyArray:list)->Tensor:
    n  = len(data.index)

    tf1 = Tensor(n,len(KeyArray)+1)

    for i in range(0,n):
        tf1.setitem(i,0,1)
        for j in range(0,len(KeyArray)):
            tf1.setitem(i,j+1,data[KeyArray[j]].array[i])
    
    return tf1
