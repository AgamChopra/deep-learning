#for 1D array input X with m batch size
#np.round(x,9)
import numpy as np

epsilon = 1E-9

def init_parm(nL,nLp,m=1): #nL is # of neurons of current layer, nLp is that of the previous layer. for first layer, nLp = input shape assuming 1D input array
    w = np.random.default_rng().standard_normal(size=(nL, nLp), dtype='float64')
    b = np.zeros((nL,m),dtype='float64')#m is the # of mini batches
    return (w,b)

def Activation(activation,z,b=1):
    if activation == 'agam':
        return (b*z)/(np.exp(z/(2*np.pi))+np.exp(-2*np.pi*z))
    elif activation == 'PReLU':
        return z*b if z < 0 else z  
    elif activation == 'swish':
        return b*z/(1 + np.exp(-z))
    elif activation == 'tanh':
        return np.tanh(z)
    elif activation == 'sigmoid':
        return (1/(1 + np.exp(-z)))
    else:
        print("Error A_fp")
        return None

def BackActivation(activation,z,b=1):
    if activation == 'agam':
        return (b/(np.exp(z/(2*np.pi))+np.exp(-2*np.pi*z)))*(1-z*(((np.exp(z/(2*np.pi))-2*np.pi*np.exp(2*np.pi*z))/(2*np.pi))/((np.exp(z/(2*np.pi))+np.exp(-2*np.pi*z)))))
    elif activation == 'PReLU':
        return b if z < 0 else 1
    elif activation == 'swish':
        return (b/(1+np.exp(-z)))*(1+z*np.exp(-z)/(1+np.exp(-z)))
    elif activation == 'tanh':
        return 1-np.square(np.tanh(z))
    elif activation == 'sigmoid':
        return (np.exp(-z)/np.square((1 + np.exp(-z))))
    else:
        print("Error A_bp")
        return None

def cross_entropy(y,aL):
    Loss = 0. #!!!needs work!!!
    return Loss

def SVM(y,aL):
    Loss = 0. #!!!needs work!!!
    return Loss

def Cost(y,aL,cost_type = 'L1'):
    if cost_type == 'cross entropy':
        return cross_entropy(y,aL)#Cross Entropy Cost
    elif cost_type == 'L2':
        return np.mean(np.square(y-aL))/2 #L2/MSE Cost
    elif cost_type =='SVM': # !!!!! need work !!!!
        return SVM(y,aL) #SVM Cost
    else:
        return np.mean(y-aL) #L1 Cost by default

def gradient_descent(w, b, dw, db, alpha, optimization = ''):
    if optimization == 'ADAM':
        return None #!!!needs work!!!
    else:
        w = w - alpha*(dw)
        b = b - alpha*(db)
        return w,b

class Layer:
    def __init__(self,nL,nLp,activation,ac_parm = 1,batch_size = 1):#initialize the neurons for layer L
        self.w,self.b = init_parm(nL, nLp, batch_size)
        self.nonLin = activation
        self.nonLin_scale = ac_parm
        self.z = self.b
        self.aL = self.b
        self.dw = np.zeros((self.w.shape))
        self.db = np.zeros((self.b.shape))
        
    def fprop(self,aLp):#forward propogate layer L. output aL = g(b+(w x aLp))
        print("\nb\n",self.b,"\nw\n",self.w,"\naLprevious\n", aLp)
        print("\nz before\n",self.z)
        self.z = self.b + np.matmul(self.w, aLp) #!!!something over here is going wrong!!!
        print("\nz after\n",self.z,"\n")
        self.aL = Activation(self.nonLin,self.z,self.nonLin_scale)
        #print(self.w.shape,aLp.shape, self.b.shape,self.z.shape, self.aL.shape)
        return self.aL
    
    def cost(self, y, aL, cost_type = 'L1'):
        self.J = Cost(y, aL, cost_type)
        return self.J
    
    def bprop(self, daL, aLp):
        if daL.size > 1:
            #print(daL.shape, BackActivation(self.nonLin, self.z, self.nonLin_scale).shape)
            self.dz = daL*BackActivation(self.nonLin, self.z, self.nonLin_scale)
        elif daL.size == 1:
            #print(daL.shape, BackActivation(self.nonLin, self.z, self.nonLin_scale).shape)
            self.dz = daL*BackActivation(self.nonLin, self.z, self.nonLin_scale)
        else:
            print("Error Layer_bp incorrect aL size")
            return None
        daLp = np.matmul(self.w.T, self.dz)
        self.db = self.dz
        self.dw = np.matmul(self.dz, aLp.T)
        #print(self.dw.shape,daL.shape,self.db.shape,daLp.shape)
        return daLp
    
    def update_parm(self, alpha = 0.01, optimization = ''):
        self.w,self.b = gradient_descent(self.w, self.b, self.dw, self.db, alpha, optimization)
    
    def show_parameters(self, show = True):
        if show:
            print("\n w =\n",self.w,"\n b =\n",self.b)
        return (self.w,self.b)
        
#%%  
np.random.seed(seed=50)    
X = np.random.rand(10,16)*400 # 10 input features with 16 training examples
Y_exp = np.zeros((1,16))
temp = np.zeros((10,16))
for i in range(10):
    for j in range(16):
        temp[i][j] = X[i][j]>200
for j in range(16):
    if np.mean(temp[:,j]) >= 0.6:
        print(np.mean(temp[:,j]))
        Y_exp[:,j] = 1
    else:
        Y_exp[:,j] = 0  
#%%
class model():
    def __init__(self,x, y):
        self.L1 = Layer(nL=3, nLp=10, activation='sigmoid', ac_parm=1, batch_size =4)
        self.L2 = Layer(nL=2, nLp=3, activation='sigmoid', ac_parm=1, batch_size =4)
        self.L3 = Layer(nL=1, nLp=2, activation='sigmoid', ac_parm=1, batch_size =4)
        self.J = []
        
    def run(self, x, y, t, learning_rate = 0.001):
        for epoch in range(t):
            print("\nEpoch:",epoch)
            for m in (4,8,12,16):
                a1 = self.L1.fprop(x[:,(m-4):(m)])
                a2 = self.L2.fprop(a1)
                a3 = self.L3.fprop(a2)
                Jeph = self.L3.cost(y[0,(m-4):(m)], a3, 'L2')
                da2 = self.L3.bprop(Jeph, a2)
                da1 = self.L2.bprop(da2, a1)
                self.L1.bprop(da1, x[:,(m-4):(m)])
                self.L3.update_parm(learning_rate)
                self.L2.update_parm(learning_rate)
                self.L1.update_parm(learning_rate)
                self.J.append(Jeph)
                print("Loss =",Jeph,"\nminibatch done\n")
        return self.J
    
    def save_model(self,):
        return self.L1.show_parameters(False),self.L1.show_parameters(False),self.L1.show_parameters(False)
#%%
test = model(X,Y_exp)
J = test.run(x = X, y = Y_exp, t = 1, learning_rate = 0.00001)
#%%
from matplotlib import pyplot as plt
plt.plot(J, color='red',linewidth=1,label = "Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss after 5 epoches')
plt.show()
#%%
at = np.random.default_rng().standard_normal(size=(3,10), dtype='float64')
print(at)
bt = at = np.random.default_rng().standard_normal(size=(10,4), dtype='float64')
print(np.matmul(at, bt.T))
