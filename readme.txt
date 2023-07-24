# BasicNeuralNetwork_CPP

this libirary is meant to be a soldification of my ideas of ai during learning it . I build this repo during studying ai and i do upgrade
it every time i learn something new. i really appriciate any comments.

THE LIBERARY
first of all there is the basic cpp code which does all the hard work , the cpp code is cpu computed , my intentions is to upgrade it to 
be gpu based computations in future. keeping every thing simple for now. the code is formed of the main class NeuralNetwork which do its 
operations using tensor math, the tensor class is made to be used by NeuralNetwork class, the tensor class is tempelated class to allow 
the chance of using other data types other than floats as doubles or complex numbers ,although only floats are used for now.
the NeuralNetwork dont normalize data and potential for digit overflow is possible and do be a problem in mean time.

second layer of code is a C set of wraper functions which calls cpp functions and export them in the dll file to allow usage outside cpp 
environment.

Third layer is python wraper which extend python with that C interface and build back the object oriented structure . class is build in python 
for NeuralNetwork and tensor to be used by the user easily .

this project is really helpful for me as it express the real understandig I have in AI and software architicture , which is not deep or wide 
but in the need can be useable.

Thank you.

