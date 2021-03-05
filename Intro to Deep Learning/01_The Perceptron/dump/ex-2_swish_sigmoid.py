# Â©[2021][Agamdeep Chopra]
# Feel free to use this code for learning purposes but please properly credit me in APA format.

import numpy as np
from matplotlib import pyplot as plt

np.random.seed(seed=65)

alpha = 0.0008
epsilon = 1E-9

X = np.random.rand(1000)*400 # input
Y_exp = np.zeros((X.shape))
for i in range(len(X)):
    Y_exp[i] = X[i]>200       # expected output

def norm_input(X):
    X = (X-np.min(X))/(np.max(X)-np.min(X))
    return X

def percp_parm():
    w = np.random.randn((1))
    b = np.zeros((1))
    return (w,b)

def loss(yp,ye):
    return (np.square(ye-yp))

def cost(yp,ye):
    return (np.mean(loss(yp,ye))/2)

def sigmoid(x):
    return (1/(1 + np.exp(-x)))

#def swish(x, b=1):
    #return b*x*sigmoid(x)
    
def swish(x, b=1):
    return (b*x)/(np.exp(x/(2*np.pi))+np.exp(-2*np.pi*x))
#def swish(x, b=1):
    #return PReLU(x,b)
#def swish(x, b=1):
    #return sigmoid(x)

def PReLU(x,b=1):
    return(x-(((x*b)>x)*(x-b*x)))

def fprop(x,w2,b2,w1,b1):
    return (sigmoid(w2*(swish(x*w1+b1))+b2))

def bprop(alpha,x,yp,ye,w2,b2,w1,b1):
    dw2,db2,dw1,db1 = diff(x, swish(x*w1+b1), yp, ye, w2, b2, w1, b1)
    w2 = w2 - alpha*dw2 
    b2 = b2 - alpha*db2
    w1 = w1 - alpha*dw1 
    b1 = b1 - alpha*db1
    return (w2,b2,w1,b1)

def diff(x,a,yp,ye,w2,b2,w1,b1):
    dJ = (loss(yp+epsilon,ye)-loss(yp-epsilon,ye))/(2*epsilon)
    dw2 = dJ*((sigmoid(a*(w2+epsilon)+b2)-sigmoid(a*(w2-epsilon)+b2))/(2*epsilon))
    db2 = dJ*((sigmoid(a*w2+(b2+epsilon))-sigmoid(a*w2+(b2-epsilon)))/(2*epsilon))
    dw1 = dJ*((sigmoid((a+epsilon)*w2+b2)-sigmoid((a-epsilon)*w2+b2))/(2*epsilon))*((swish(x*(w1+epsilon)+b1)-swish(x*(w1-epsilon)+b1))/(2*epsilon))
    db1 = dJ*((sigmoid((a+epsilon)*w2+b2)-sigmoid((a-epsilon)*w2+b2))/(2*epsilon))*((swish(x*w1+(b1+epsilon))-swish(x*w1+(b1-epsilon)))/(2*epsilon))
    return (dw2,db2,dw1,db1)

def perceptron(x,ye,epoch=500): 
    w1,b1 = percp_parm()
    w2,b2 = percp_parm()
    ls = []
    list_index = 0
    x = norm_input(x) # Normalizing the input. This is done to prevent exploding or vanishing gradients.
    for i in range(epoch):
        for j in range(len(x)):
            yp = fprop(x[j], w2, b2, w1, b1)
            w2,b2,w1,b1 = bprop(alpha, x[j], yp, ye[j], w2, b2, w1, b1)
            if i%20 == 0 and j == 1:
                ls.append(loss(yp, ye[j]))
                print("Epoch:",i," Loss:", ls[list_index])
                list_index += 1
    return w2,b2,w1,b1,ls
#%%
#Run the following:
w2,b2,w1,b1,J = perceptron(X, Y_exp,epoch = 500)
print("optimized weight[2,1] = ",w2,w1," and bias[2,1] = ",b2,b1)
#%%
Jrlu = J
#%%
Jagam = J
#%%
Jswish = J
#%%
Jsig = J
#%%
plt.plot(Jsig, color='black',linewidth=1,label = "sigmoid-sigmoid")
plt.plot(Jrlu, color='blue',linewidth=1,label = "PReLU-sigmoid")
plt.plot(Jswish, color='green',linewidth=1,label = "swish-sigmoid")
plt.plot(Jagam, color='red',linewidth=1,label = "(bx/(e^(x/2π)+e^(-2πx)))-sigmoid")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss after 500 epoches')
plt.legend()
plt.show()
#%%
print("\nTesting:")
Xt = np.random.rand(100)*400
Yt = np.zeros((Xt.shape))
for i in range(len(Xt)):
    Yt[i] = Xt[i]>200
#print("\nTest Set:\n",Xt,"\nPrediction:\n",np.around(fprop(norm_input(Xt),w2,b2,w1,b1)),"\nExpectation:\n",Yt)
tt=np.around(fprop(norm_input(Xt),w2,b2,w1,b1))
for i in range(len(Xt)):
    tt[i] = tt[i]==Yt[i]
#print(tt)
print("Test Accuracy=", np.mean(tt))
# Expected output for seed=65, epoch = 400 -> Test accuracy ≈ 1.0
#%%
'''
If everything works as expected, you should get something like this:
Epoch: 0  Loss: [0.26148347]
Epoch: 20  Loss: [0.20311161]
...
Epoch: 360  Loss: [2.47054208e-05]
Epoch: 380  Loss: [1.7035298e-05]
optimized weight[2,1] =  [-4.68616377] [-4.34582901]  and bias[2,1] =  [2.33193864] [2.9072655]

Testing:

Test Set:
 [121.35055235  65.180089   279.72550129 308.60433267 205.82097905
  94.32275525  92.01585064 202.28606654  53.67398519 224.58490519
 ...
 391.87359994 202.83060501 318.4418724  160.97708227  69.98939625
 262.00612419 104.73410931 152.34305253 255.62472255 342.85440107] 
Prediction:
 [0. 0. 1. 1. 1. 0. 0. 1. 0. 1. 0. 1. 1. 0. 0. 1. 0. 0. 1. 0. 0. 0. 1. 0.
 0. 1. 1. 1. 1. 0. 0. 0. 0. 1. 1. 0. 1. 0. 0. 0. 0. 0. 1. 0. 1. 1. 0. 0.
 1. 1. 0. 1. 1. 0. 1. 0. 1. 1. 1. 0. 0. 0. 1. 1. 0. 0. 0. 1. 0. 0. 1. 0.
 1. 0. 1. 1. 1. 0. 1. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0. 1.
 0. 0. 1. 1.] 
Expectation:
 [0. 0. 1. 1. 1. 0. 0. 1. 0. 1. 0. 1. 1. 0. 0. 1. 0. 0. 1. 0. 0. 0. 1. 0.
 0. 1. 1. 1. 1. 0. 0. 0. 0. 1. 1. 0. 1. 0. 0. 0. 0. 0. 1. 0. 1. 1. 0. 0.
 1. 1. 0. 1. 1. 0. 1. 0. 1. 1. 1. 0. 0. 0. 1. 1. 0. 0. 0. 1. 0. 0. 1. 0.
 1. 0. 1. 1. 1. 0. 1. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0. 1.
 0. 0. 1. 1.]
[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1.]
Test Accuracy= 1.0

Your accuracy might be very different depending on the random initializations. 
If accuracy is very bad, try running the code again and perhapse use random seed to have more control over the rand initializations.
'''