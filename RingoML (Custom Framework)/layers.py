#for 1D array input X
import numpy as np

epsilon = 1E-9

def init_parm(nL,nLp,m=1): #nL is # of neurons of current layer, nLp is that of the previous layer. for first layer, nLp = input shape assuming 1D input array
    w = np.random.randn((nL, nLp))
    b = np.zeros((nL,m))#m is the # of mini batches
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

class Layer:
    def __init__(self,nL,nLp,activation,ac_parm = 1,batch_size = 1):#initialize the neurons for layer L
        self.w,self.b = init_parm(nL, nLp, batch_size)
        self.nonLin = activation
        self.nonLin_scale = ac_parm
        self.z = self.b
        self.aL = self.b
        
    def fprop(self,aLp):#forward propogate layer L. output aL = g(b+(w x aLp))
        self.z = self.b + np.cross(self.w, aLp)
        self.aL = Activation(self.z,self.nonLin,self.nonLin_scale)
        return self.aL
    
    def bprop(self,Layer_Final = False):
        if Layer_Final:
            
            return
        else:
            
            return self.w,self.b
        
        
#%%      
AL = np.random.rand(100,10) #x,y,z,masks,batches
Y = np.zeros((100,10))

print(AL.shape)
print(Y.shape)
print(Cost(Y, AL, 'L2'))
print(np.square(Y-AL).shape)
