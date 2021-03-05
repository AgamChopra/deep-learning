# ©[2021][Agamdeep Chopra]
# Feel free to use this code for learning purposes but please properly credit me in APA format.
	# Example 1, Perceptron Binary Inverter
	# Suppliment to my Introduction to Deep Learning Video 1
	# (A simple binary inverter using a perceptron logic.)
	# Again, as mentioned in the video, this code can easily be outperformed-
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
alpha = 0.0008 #learning rate. Dont worry aboput this for now.
eta = 1E-9 #very small value constant. Dont worry aboput this for now.

#Define the inputs and ground truth values(expected outputs)
X = np.random.rand(1000)*400 # input
Y_exp = np.zeros((X.shape))
for i in range(len(X)):
    Y_exp[i] = X[i]>200 # expected output
def norm_input(X):
    X = (X-np.min(X))/(np.max(X)-np.min(X))
    return X
#initialize the parameters
def percp_parm():
    w = np.random.randn((1)) #initialize weight value of perseptron to some random value. Dont worry about this for now.
    b = np.zeros((1))  #initialize bias value of perseptron as zero. Dont worry about this for now.
    return (w,b)

#Define the loss. Dont worry about loss or cost functions. 
#For this example we only use a simple mean square error loss.
def loss(yp,ye):
    return (np.square(ye-yp))

#Dont worry about cost for now. We wont use it for this example.
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
def bprop(alpha,x,yp,ye,w,b):
    dw,db = diff(x, yp, ye, w, b) #calculate dx and dy
    w = w - alpha*dw #update weight
    b = b - alpha*db #update bias
    #print("w back",w.shape,"b back",b.shape)
    return (w,b)

#Calculate the delta values using chain rule on Loss J and partial derivatives.
def diff(x,yp,ye,w,b):
    dw = ((loss(yp+eta,ye)-loss(yp-eta,ye))/(eta))*((fprop(x,w+eta,b)-fprop(x,w-eta,b))/(2*eta))
    # dw <- dJ/dw = (dJ/dy)(dy/dw)
    #             = ((J(y+eta)-J(y-eta))/2eta)*((y(w+eta)-y(w-eta))/2eta) {where y = actiation(linear(x,w,b))
    #                                                                            J = Loss(y,y_exp)           }
    db = ((loss(yp+eta,ye)-loss(yp-eta,ye))/(eta))*((fprop(x,w,b+eta)-fprop(x,w,b-eta))/(2*eta))
    # db <- dJ/db = (dJ/dy)(dy/db)
    #             = ((J(y+eta)-J(y-eta))/2eta)*((y(b+eta)-y(b-eta))/2eta) {where y = actiation(linear(x,w,b))
    #                                                                            J = Loss(y,y_exp)           }
    #print("dw diff",dw.shape,"db diff",db.shape)
    return (dw,db)

#Putting it all together in the Perceptron model
#Epoch is the number of times we want to run the model with backprop to minimize J.
def perceptron(x,ye,epoch=100): 
    w,b = percp_parm() #initializing the parameters
    #print("w initial ",w.shape,"b initial ",b.shape)
    ls = []
    list_index = 0
    x=norm_input(x)
    for i in range(epoch):#loop to minimize J.
        for j in range(len(x)):#in future, we will use mini batches and vectorization properties of numpy
            yp = fprop(x[j], w, b)#the forward step to calculate the output
            #print("yp fprop after ",yp.shape)
            w,b = bprop(alpha, x[j], yp, ye[j], w, b)#update the parameters.
            #print("w after ",w.shape,"b after ",b.shape)
            #loss calculation for visualization. Dont worry about the next 3 lines
            if i%20 == 0 and j == 1:
                ls.append(loss(yp, ye[j]))
                print("Epoch:",i," Loss:", ls[list_index])
                list_index += 1
    return w,b,ls
#%%
#Finally! We are all set!
#Lets try running the model.
#If everything works properly, you should see something like:
#[0.18596151]
#[0.12121284]
#....
#[0.00908715]
#[0.00857715]
#optimized weight =  [-4.58871226]  and bias =  [2.17494229]
np.random.seed(seed=65)
w,b,J = perceptron(X, Y_exp,epoch = 400)
print("optimized weight = ",w," and bias = ",b)
#%%
#testing the solution
#here, as long as our model returns a value larger than 0.5, we say that the prediction is 1 and vice versa.
plt.plot(J, color='red',linewidth=1,label = "Loss")
plt.xlabel('Epoch x20')
plt.ylabel('Loss')
plt.title('(Sigmoid)')
plt.show()
print("\nTesting:")
Xt = np.random.rand(100)*400
Yt = np.zeros((Xt.shape))
for i in range(len(Xt)):
    Yt[i] = Xt[i]>200
print("\nTest Set:\n",Xt,"\nPrediction:\n",np.around(fprop(norm_input(Xt),w,b)),"\nExpectation:\n",Yt)
tt=np.around(fprop(norm_input(Xt),w,b))
for i in range(len(Xt)):
    tt[i] = tt[i]==Yt[i]
print(tt)
print("Test Accuracy=", np.mean(tt))