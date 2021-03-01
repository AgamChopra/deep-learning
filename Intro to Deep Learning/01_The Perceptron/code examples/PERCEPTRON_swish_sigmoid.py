import numpy as np
from matplotlib import pyplot as plt

alpha = 0.001
epsilon = 1E-9

X = np.array((166,177,190,178,148,159,214,165,186,196,168,171,217,191.204,222,213,276,213,193,212,229,215,260,258,266,331)) # input
X = X/np.linalg.norm(X)
Y_exp = np.array((0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1)) # expected output

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

def swish(x, b=1):
    return b*x*sigmoid(x)

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

def perceptron(x,ye,epoch=100): 
    w1,b1 = percp_parm()
    w2,b2 = percp_parm()
    ls = []
    list_index = 0
    for i in range(epoch):
        for j in range(len(x)):
            yp = fprop(x[j], w2, b2, w1, b1)
            w2,b2,w1,b1 = bprop(alpha, x[j], yp, ye[j], w2, b2, w1, b1)
            if i%100 == 0 and j == 1:
                ls.append(loss(yp, ye[j]))
                print("Loss", ls[list_index])
                list_index += 1
    return w2,b2,w1,b1,ls
#%%
#Run the following:
w2,b2,w1,b1,J = perceptron(X, Y_exp,epoch = 1500)
print("optimized weight[2,1] = ",w2,w1," and bias[2,1] = ",b2,b1)
plt.plot(J, color='red',linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('(Swish->Sigmoid)')
plt.show()
print(X,"\n-o->\n",np.around(fprop(X,w2,b2,w1,b1)),"\n?=\n",Y_exp)
tt=np.around(fprop(X,w2,b2,w1,b1))
for i in range(len(Y_exp)-1):
    tt[i] = tt[i]==Y_exp[i]
print(tt)
print("Accuracy=", np.mean(tt))
#%%
'''
If everything works as expected, you should get something like this:
Loss [0.43256865]
Loss [0.27795202]
Loss [0.24842286]
Loss [0.24105851]
Loss [0.2386457]
Loss [0.23738767]
Loss [0.23638189]
Loss [0.23541746]
Loss [0.23444735]
Loss [0.23346188]
Loss [0.23245947]
Loss [0.23144021]
Loss [0.23040453]
Loss [0.22935289]
Loss [0.22828577]
optimized weight[2,1] =  [3.13696302] [2.77278315]  and bias[2,1] =  [-0.55785115] [-0.19068793]

 [0.15330087 0.16345937 0.17546486 0.16438286 0.13667789 0.14683638
 0.19762884 0.15237737 0.17177086 0.18100585 0.15514787 0.15791837
 0.20039934 0.17657675 0.20501683 0.19670534 0.25488579 0.19670534
 0.17823535 0.19578184 0.21148133 0.19855234 0.2401098  0.2382628
 0.2456508  0.30567825] 
-o->
 [0. 0. 0. 0. 0. 0. 1. 0. 0. 1. 0. 0. 1. 0. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1.
 1. 1.] 
?=
 [0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1]
[1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 1. 1. 1. 1.
 1. 1.]
Accuracy= 0.8846153846153846

Your accuracy might be very different depending on the random initializations. 
If accuracy is very bad, try running the code again and perhapse use random seed to have more control over the rand initializations.
'''