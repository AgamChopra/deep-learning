#for 1D array input X with m batch size
#np.round(x,9)
#np.clip(in_array, a_min = -100, a_max = 100) 
import numpy as np

epsilon = 1E-9
clipmax = 500.0
clipmin = -clipmax
precision = 20

def init_parm(nL,nLp,m=1): #nL is # of neurons of current layer, nLp is that of the previous layer. for first layer, nLp = input shape assuming 1D input array
    w = np.round(np.random.default_rng().standard_normal(size=(nL, nLp), dtype='float64'),9)
    b = np.zeros((nL,m),dtype='float64')#m is the # of mini batches
    return (w,b)

def max_norm(x):
    return (x/np.max(x))

def Activation(activation,z,b=1):
    z = np.round(np.clip(z,clipmin,clipmax),precision)
    if activation == 'agam':
        return (b*z)/(np.exp(z/(2*np.pi))+np.exp(-2*np.pi*z))
    elif activation == 'PReLU':
        return (z-(((z*b)>z)*(z-b*z)))  
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
    z = np.round(np.clip(z,clipmin,clipmax),precision)
    if activation == 'agam':
        return (b/(np.exp(z/(2*np.pi))+np.exp(-2*np.pi*z)))*(1-z*(((np.exp(z/(2*np.pi))-2*np.pi*np.exp(2*np.pi*z))/(2*np.pi))/((np.exp(z/(2*np.pi))+np.exp(-2*np.pi*z)))))
    elif activation == 'PReLU':
        return (Activation('PReLU',z+epsilon,b)-Activation('PReLU',z-epsilon,b))/(2*epsilon) 
    elif activation == 'swish':
        return (b/(1+np.exp(-z)))*(1+z*np.exp(-z)/(1+np.exp(-z)))
    elif activation == 'tanh':
        return 1-np.square(np.tanh(z))
    elif activation == 'sigmoid':
        return (np.exp(-z)/np.square((1 + np.exp(-z))))
    else:
        print("Error A_bp")
        return None

#https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
def Cost(y,aL,cost_type = 'L1'):
    y = np.round(y,precision)
    aL = np.round(aL,precision)
    if cost_type == 'cross entropy':
        return np.round(np.mean(((y*np.log(aL)+(1-y)*(1-np.log(aL))))),precision)
    elif cost_type == 'L2':
        return np.round(np.mean(np.square(y-aL))/2,precision) #L2/MSE Cost
    elif cost_type =='SVM':
        return  #SVM Cost
    else:
        return np.round(aL,np.mean(np.abs(y-aL)),precision) #L1 Cost by default
    
def Cost_back(y,aL,cost_type = 'L1'):
    y = np.round(y,precision)
    aL = np.round(aL,precision)
    if cost_type == 'cross entropy':
        return np.round(np.mean(((y-aL)/(1E-20+aL*(1-aL)))),precision)
    elif cost_type == 'L2':
        return np.round(np.mean(y-aL),precision)
    elif cost_type =='SVM':
        return  #SVM Cost
    else:
        return -1 if np.round(np.mean(y-aL),precision)<0. else 1
    
def gradient_descent(w, b, dw, db, alpha, optimization = ''):
    '''
    w = np.round(np.clip(w,clipmin,clipmax),precision)
    dw = np.round(np.clip(dw,clipmin,clipmax),precision)
    b = np.round(np.clip(b,clipmin,clipmax),precision)
    db = np.round(np.clip(db,clipmin,clipmax),precision)
    '''
    if optimization == 'ADAM':
        return None #!!!needs work!!!
    else:
        w = np.round(w - alpha*dw,precision)
        b = np.round(b - alpha*db,precision)
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
        #print("\nb\n",self.b,"\nw\n",self.w,"\naLprevious\n", aLp)
        #print("\nz before\n",self.z)
        self.z = np.round(self.b + np.matmul(self.w, max_norm(aLp)),precision)
        #print("\nz after\n",self.z,"\n")
        self.aL = np.round(Activation(self.nonLin,self.z,self.nonLin_scale),precision)
        #print(self.w.shape,aLp.shape, self.b.shape,self.z.shape, self.aL.shape)
        return self.aL
    
    def cost(self, y, cost_type = 'L1'):
        self.J = Cost(y, self.aL, cost_type)
        return self.J
    
    def cost_back(self, y, cost_type = 'L1'):
        return Cost_back(y, self.aL, cost_type)
    
    def bprop(self, daL, aLp):
        self.dz = daL*BackActivation(self.nonLin, self.z, self.nonLin_scale)
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
        return (self.w,np.mean(self.b,axis=1))
        
#%%  
np.random.seed(seed=50)    
X = np.random.rand(10,10)*400 # 10 input features with 16 training examples
Y_exp = np.zeros((1,10))
temp = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        temp[i][j] = X[i][j]>200
for j in range(10):
    if np.mean(temp[:,j]) >= 0.6:
        print(np.mean(temp[:,j]))
        Y_exp[:,j] = 1
    else:
        Y_exp[:,j] = 0 
X = X / X.max(axis=0)
#%%
class model():
    def __init__(self,x, y):
        self.L1 = Layer(nL=20, nLp=10, activation='PReLU', ac_parm=0.1, batch_size =10)
        self.L2 = Layer(nL=10, nLp=20, activation='PReLU', ac_parm=0.1, batch_size =10)
        self.L3 = Layer(nL=10, nLp=10, activation='PReLU', ac_parm=0.1, batch_size =10)
        self.L4 = Layer(nL=5, nLp=10, activation='PReLU', ac_parm=0.1, batch_size =10)
        self.L5 = Layer(nL=5, nLp=5, activation='PReLU', ac_parm=0.1, batch_size =10)
        self.L6 = Layer(nL=4, nLp=5, activation='PReLU', ac_parm=0.1, batch_size =10)
        self.L7 = Layer(nL=3, nLp=4, activation='PReLU', ac_parm=0.1, batch_size =10)
        self.L8 = Layer(nL=3, nLp=3, activation='PReLU', ac_parm=0.1, batch_size =10)
        self.L9 = Layer(nL=1, nLp=3, activation='sigmoid', ac_parm=1, batch_size =10)
        self.J = []
        
    def run(self, x, y, t, learning_rate = 0.001,decay = True):
        for epoch in range(t):
            #print("\nEpoch:",epoch)
            #for m in (16):
            learning_rate_decay = learning_rate/(epoch+1) if decay else learning_rate
            m=10   
            a1 = self.L1.fprop(x[:,(m-10):(m)])
            a2 = self.L2.fprop(a1)
            a3 = self.L3.fprop(a2)
            a4 = self.L4.fprop(a3)
            a5 = self.L5.fprop(a4)
            a6 = self.L6.fprop(a5)
            a7 = self.L7.fprop(a6)
            a8 = self.L8.fprop(a7)
            self.L9.fprop(a8)
            Jeph = self.L9.cost(y[0,(m-10):(m)], 'L2')
            da9 = self.L9.cost_back(y[0,(m-10):(m)], 'L2')
            da8 = self.L3.bprop(da9, a8)
            da7 = self.L3.bprop(da8, a7)
            da6 = self.L3.bprop(da7, a6)
            da5 = self.L3.bprop(da6, a5)
            da4 = self.L3.bprop(da5, a4)
            da3 = self.L3.bprop(da4, a3)
            da2 = self.L3.bprop(da3, a2)
            da1 = self.L2.bprop(da2, a1)
            self.L1.bprop(da1, x[:,(m-10):(m)])
            self.L9.update_parm(learning_rate_decay)
            self.L8.update_parm(learning_rate_decay)
            self.L7.update_parm(learning_rate_decay)
            self.L6.update_parm(learning_rate_decay)
            self.L5.update_parm(learning_rate_decay)
            self.L4.update_parm(learning_rate_decay)
            self.L3.update_parm(learning_rate_decay)
            self.L2.update_parm(learning_rate_decay)
            self.L1.update_parm(learning_rate_decay)
            if epoch % 50 == 0 or epoch == 0:
                self.J.append(Jeph)
            if epoch % 100 ==0 or epoch == 0:
                print("Loss after",epoch,"epochs =", Jeph)
            #print("Loss =",Jeph,"\nminibatch done\n")
        return self.J
    '''
    def save_model(self,):
        return (self.L1.show_parameters(False),self.L2.show_parameters(False),self.L3.show_parameters(False))
    
    def test_model(self,x,w1,w2,w3,w4,w5,w6,w7,w8,w9,b1,b2,b3,b4,b5,b6,b7,b8,b9):
        L1 = Layer(nL=100, nLp=10, activation='PReLU', ac_parm=0.1, batch_size =1)
        L2 = Layer(nL=50, nLp=100, activation='PReLU', ac_parm=0.1, batch_size =1)
        L3 = Layer(nL=50, nLp=50, activation='PReLU', ac_parm=0.1, batch_size =1)
        L4 = Layer(nL=20, nLp=50, activation='PReLU', ac_parm=0.1, batch_size =1)
        L5 = Layer(nL=20, nLp=20, activation='PReLU', ac_parm=0.1, batch_size =1)
        L6 = Layer(nL=10, nLp=20, activation='PReLU', ac_parm=0.1, batch_size =1)
        L7 = Layer(nL=5, nLp=10, activation='PReLU', ac_parm=0.1, batch_size =1)
        L8 = Layer(nL=5, nLp=5, activation='PReLU', ac_parm=0.1, batch_size =1)
        L9 = Layer(nL=1, nLp=5, activation='sigmoid', ac_parm=1, batch_size =1)
        L1.w = w1
        L1.b = b1
        L2.w = w2
        L2.b = b2
        L3.w = w3
        L3.b = b3
        L4.w = w4
        L4.b = b4
        L5.w = w5
        L5.b = b5
        L6.w = w6
        L6.b = b6
        L7.w = w7
        L7.b = b7
        L8.w = w8
        L8.b = b8
        L9.w = w9
        L9.b = b9
        a1 = L1.fprop(x)
        a2 = L2.fprop(a1)
        a3 = L3.fprop(a2)
        a4 = L3.fprop(a3)
        a5 = L3.fprop(a4)
        a6 = L3.fprop(a5)
        a7 = L3.fprop(a6)
        a8 = L3.fprop(a7)
        a9 = L3.fprop(a8)
        return a9
        '''
#%%
test = model(X,Y_exp)
J = test.run(x = X, y = Y_exp, t = 5000, learning_rate = 1E-8, decay = True)

from matplotlib import pyplot as plt
plt.plot(J, color='red',linewidth=1,label = "ReLU in hidden layers accuracy = 0.65")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss after n epoches')
plt.show()
#%%