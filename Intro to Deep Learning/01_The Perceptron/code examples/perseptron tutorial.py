# ©[2021][Agamdeep Chopra]
# Feel free to use this code for learning purposes but please properly credit me in APA format.
# Example 1, Perceptron Binary Inverter
# Suppliment to my Introdution to Deep Learning Video 1
# (A simple binary inverter using a perceptron logic.)
# Again, as mentioned in the video, this code can easily be outperfomed-
# by a simple inversion logic. The goal of this project is to give a-
# a gentle introduction to ML.

#import these files
#uncomment the pip commands and run them if numpy or matplotlib are not correctly installed.
#pip install numpy
#pip install matplotlib
import numpy as np
from matplotlib import pyplot as plt

#%%
#Define the hyperparameters
η = 0.001 #learning rate. Dont worry aboput this for now.
ε = 1E-9 #very small value constant. Dont worry aboput this for now.

#Define the inputs and ground truth values(expected outputs)
X = np.array((1,0)) # input
Y_exp = np.array((0,1)) # expected output

#initialize the parameters
def percp_parm():
    w = np.random.randn((1)) #initialize weight value of perseptron to some random value. Dont worry about this for now.
    b = np.zeros((1))  #initialize bias value of perseptron as zero. Dont worry about this for now.
    return (w,b)

#Define the loss. Dont worry about loss or cost functions. 
#For this example we only use a simple mean square error loss.
def loss(yp,ye):
    return (np.square(ye-yp))

#We sont have any need to use cost for this example but just to give an example.
def cost(yp,ye):
    return (np.mean(loss(yp,ye))/2)

#Activation function (non-linear part of the perceptron)
def sigmoid(x):
    return (1/(1 + np.exp(-x)))

#Forward propogation through the perceptron
def fprop(x,w,b):
    return (sigmoid(x*w+b))#applying the non-linearity over the linear equation containing the parameters w & b

#Back propogation through the perceptron
#This is how the parameters are optimized.The goal is to minimize the Cost function J-
# by ever so slightly changing the values of the parameters in the direction of the-
# minima (local or global).
#Do not worry about optimizing the gradient decent process for now.
def bprop(η,x,yp,ye,w,b):
    Δw,Δb = diff(x,yp,ye,w,b) #calculate Δx and Δy
    w_new = w - η*Δw #update weight
    b_new = b - η*Δb #update bias
    return (w_new,b_new)

#Calculate the Δ values using chain rule on Loss J and partial derivatives.
def diff(x,yp,ye,w,b):
    Δw = ((cost(yp+ε,ye)-cost(yp-ε,ye))/(2*ε))*((fprop(x,w+ε,b)-fprop(x,w-ε,b))/(2*ε))
    # Δw <- ∂J/∂w = (∂J/∂y)(∂y/∂w)
    #             = ((J(y+ε)-J(y-ε))/2ε)*((y(w+ε)-y(w-ε))/2ε) {where y = actiation(linear(x,w,b))
    #                                                                J = Loss(y,y_exp)           }
    Δb = ((cost(yp+ε,ye)-cost(yp-ε,ye))/(2*ε))*((fprop(x,w,b+ε)-fprop(x,w,b-ε))/(2*ε))
    # Δb <- ∂J/∂b = (∂J/∂y)(∂y/∂b)
    #             = ((J(y+ε)-J(y-ε))/2ε)*((y(b+ε)-y(b-ε))/2ε) {where y = actiation(linear(x,w,b))
    #                                                                J = Loss(y,y_exp)           }
    return (Δw,Δb)

#Putting it all together in the Perceptron model
#Epoch is the number of times we want to run the model with backprop to minimize J.
def perceptron(x,ye,epoch=100): 
    w,b = percp_parm() #initializing the parameters
    ls = []
    list_index = 0
    for i in range(epoch):#loop to minimize J.
        #Using vectorization properties of numpy:
        yp = fprop(x, w, b)#the forward step to calculate the output
        w,b = bprop(η, x, yp, ye, w, b)#update the parameters.
        #loss calculation for visualization. Dont worry about the next 3 lines
        if i%500 == 0:
            ls.append(cost(yp, ye))
            print(ls[list_index])
            list_index += 1
    return w,b,ls
#%%
#Finally! We are all set!
#Lets try running the model.
#If everything works properly, you should see something like:
#[-0.62291858] [0.]
#[0.18596151]
#[0.12121284]
#....
#[0.00908715]
#[0.00857715]
#optimized weight =  [-4.58871226]  and bias =  [2.17494229]
#if x = 0 then y_pred =  [1.]  with value [0.89797664]  at threshold 0.5
#if x = 1 then y_pred =  [0.]  with value [0.08212868]  at threshold 0.5
#here, as long as our model spits out a value larger than 1, we say that the prediction is 1 and vice versa.
w,b = perceptron(X, Y_exp,epoch = 10000) 
print("optimized weight = ",w," and bias = ",b)
#testing the solution
print("if x = 0 then y_pred = ",np.around(fprop(0, w, b))," with value",fprop(0, w, b)," at threshold 0.5")
print("if x = 1 then y_pred = ",np.around(fprop(1, w, b))," with value",fprop(1, w, b)," at threshold 0.5")
#%%
#pre-trained model parameters and loss for visualization.
ws,bs = [-4.54522017,2.15276728] #saved weights
print("saved weight = ",w," and bias = ",b)
print("if x = 0 then y_pred = ",np.around(fprop(0, ws, bs))," with value",fprop(0, ws, bs)," at threshold 0.5")
print("if x = 1 then y_pred = ",np.around(fprop(1, ws, bs))," with value",fprop(1, ws, bs)," at threshold 0.5")
y = np.array((0.27576416,0.0449583,0.02019229,0.01250398,0.00892285))
x =  np.array((0,25000,50000,75000,100000))
plt.plot(x, y, color='red', marker='o', linestyle='dashed',linewidth=1, markersize=4, label='output w=-4.54522017 ,b=2.15276728')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Loss of Binary I/O Perceptron during training')
plt.legend()
plt.show()
#I really hope that this introduction was informative and fun!
#Hope you're as excited as me to work on much more complex problems in future tutorials! :)
#Have a nice day and see you soon!
