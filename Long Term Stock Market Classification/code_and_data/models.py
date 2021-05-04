# models.py
# Author: Agamdeep S. Chopra, bit.ly/AgamChopra
# Last Updated: 05/01/2021

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt 
import matplotlib.colors as mcolors

Ic_global = 196

def dense(in_c, out_c):
    out = nn.Sequential(nn.Linear(in_c,out_c),nn.ReLU(out_c))
    return out

def dense_out(in_c,out_c):
    out = nn.Sequential(nn.Linear(in_c,out_c),nn.Sigmoid())
    return out

class Feedforward(nn.Module):
    def __init__(self, in_size, out_size):
        super(Feedforward, self).__init__()
        self.f = dense_out(in_size, out_size)

    def forward(self, x):
        y = self.f(x)
        return y

class Shallow_Feedforward(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super(Shallow_Feedforward, self).__init__()
        self.f1 = dense(in_size, hid_size)
        self.f2 = dense_out(hid_size,out_size)

    def forward(self, x):
        y = self.f1(x)
        y = self.f2(y)
        return y
    
class Deep_Feedforward(nn.Module):
    def __init__(self, in_size, hid1_size, hid2_size, hid3_size, hid4_size, hid5_size, out_size):
        super(Deep_Feedforward, self).__init__()
        self.f1 = dense(in_size, hid1_size)
        self.f2 = dense(hid1_size, hid2_size)
        self.f3 = dense(hid2_size, hid3_size)
        self.f4 = dense(hid3_size, hid4_size)
        self.f5 = dense(hid4_size, hid5_size)
        self.f6 = dense_out(hid5_size,out_size)
        
    def forward(self, x):
        y = self.f1(x)
        y = self.f2(y)
        y = self.f3(y)
        y = self.f4(y)
        y = self.f5(y)
        y = self.f6(y)
        return y

class Linear_Regression(nn.Module):
    def __init__(self, in_size, out_size):
        super(Linear_Regression, self).__init__()
        self.f = nn.Linear(in_size,out_size)
        
    def forward(self, x):
        y = self.f(x)
        return y
    
class Pseudo_RNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(Pseudo_RNN, self).__init__()
        self.f1 = dense(input_size, output_size)
        self.f2 = dense(input_size + output_size, output_size)
        self.f3 = dense(input_size + output_size, output_size)
        self.f4 = dense(input_size + output_size, output_size)
        self.f5 = dense(input_size + output_size, output_size)
        self.f6 = dense_out(input_size + output_size,output_size)
        
    def forward(self, x):
        y = self.f1(x)
        y = self.f2(torch.cat((x, y), -1))
        y = self.f3(torch.cat((x, y), -1))
        y = self.f4(torch.cat((x, y), -1))
        y = self.f5(torch.cat((x, y), -1))
        y = self.f6(torch.cat((x, y), -1))
        return y

def rnn(in_c):
    out = nn.Sequential(nn.Linear(in_c,6),nn.ReLU(6),nn.Linear(6,1),nn.ReLU(1))
    return out

def rnf(in_c):
    out = nn.Sequential(nn.Linear(in_c,6),nn.ReLU(6),nn.Linear(6,1),nn.Sigmoid())
    return out
 
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rn = []
        for i in range(0,Ic_global-1):
            self.rn.append(rnn(2))
        self.rnf = rnf(2)
        self.flag = False
        
    def forward(self, x):
        if(x.numpy().ndim == 2):
            x = torch.reshape(x,[1,x.numpy()[:,0].size,x.numpy()[0,:].size]) #[:,:] -> [1,:,:]
            self.flag = True
        else:
            self.flag = False
        y = torch.reshape(x[:,:,0], [x.numpy()[:,0,0].size,x.numpy()[0,:,0].size,1])
        for i in range(0,Ic_global-1):
            y = self.rn[i](torch.cat((torch.reshape(x[:,:,i], [x.numpy()[:,0,0].size,x.numpy()[0,:,0].size,1]), y), -1))
        y = self.rnf(torch.cat((torch.reshape(x[:,:,Ic_global-1], [x.numpy()[:,0,0].size,x.numpy()[0,:,0].size,1]), y), -1))
        if self.flag:
            y = y[0,:,:] #[1,:,:] -> [:,:]
        return y
    
def conv3_layer(in_c, out_c):
    conv = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size = 3),nn.BatchNorm2d(out_c),nn.ReLU(inplace=True))
    return conv

def conv2_layer(in_c, out_c):
    conv = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size = 2),nn.BatchNorm2d(out_c),nn.ReLU(inplace=True))
    return conv

def conv5_layer(in_c, out_c):
    conv = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size = 5),nn.BatchNorm2d(out_c),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=3, stride=1))
    return conv

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.f1 = conv5_layer(1, 2)
        self.f2 = conv5_layer(2, 4)
        self.f3 = conv2_layer(4, 8)
        self.f4 = dense(8,4)
        self.f5 = dense_out(4,1)
        self.flag = False
        
    def forward(self, x):
        if(x.numpy().ndim == 2):
            x = torch.reshape(x,[1,x.numpy()[:,0].size,x.numpy()[0,:].size])
            self.flag = True
        else:
            self.flag = False
        x = torch.reshape(x,[len(x[:,0,0])*len(x[0,:,0]),1,len(x[0,0,:])])
        x = torch.reshape(x,[len(x[:,0,0]),len(x[0,:,0]),14,14])
        y = self.f1(x)
        y = self.f2(y)
        y = self.f3(y)
        y = torch.flatten(y, start_dim=1, end_dim=-1)
        y = self.f4(y)
        y = self.f5(y)
        return y
    
class CNN_RNN(nn.Module):
    def __init__(self):
        super(CNN_RNN, self).__init__()
        self.f1 = conv5_layer(1, 2)
        self.f2 = conv5_layer(2, 4)
        self.f3 = conv2_layer(4, 8)
        self.rn = []
        for i in range(0,7):
            self.rn.append(rnn(2))
        self.rnf = rnf(2)
        self.flag = False
        
    def forward(self, x):
        if(x.numpy().ndim == 2):
            x = torch.reshape(x,[1,x.numpy()[:,0].size,x.numpy()[0,:].size])
            self.flag = True
        else:
            self.flag = False
        #print(x.shape)
        x = torch.reshape(x,[len(x[:,0,0])*len(x[0,:,0]),1,len(x[0,0,:])])
        x = torch.reshape(x,[len(x[:,0,0]),len(x[0,:,0]),14,14])
        #print(x.shape)
        y = self.f1(x)
        y = self.f2(y)
        y = self.f3(y)
        x = torch.flatten(y, start_dim=1, end_dim=-1)
        print(x.shape)
        y = torch.reshape(x[:,0],[len(x[:,0]),1])
        for i in range(0,6):
            y = self.rn[i](torch.cat((torch.reshape(x[:,i],[len(x[:,0]),1]), y), -1))
        y = self.rn[6](torch.cat((torch.reshape(x[:,6],[len(x[:,0]),1]), y), -1))
        #if self.flag:
            #y = y[0,:,:]
        return y      
   
def calc_acc(y_pred,y_exp):#accuracy. higher is better
    accuracy = 0.0
    for i in range(len(y_pred)):
        if int(y_pred[i]) == int(y_exp[i]):
            accuracy += 1 
    return accuracy/len(y_pred)

def calc_FPR(y_pred,y_exp):#false positive rate. lower is better
    FP=0.0
    TN=0.0
    for i in range(len(y_pred)):
        if(int(y_pred[i])==1 and int(y_exp[i]) != 1):
            FP+=1
        elif(int(y_pred[i])==0 and int(y_exp[i]) == 0):
            TN+=1
    FPR = FP/(FP+TN)
    return FPR

def fixed_pred(y_p, optm = 0.5, rev=False):
    if(not(rev)):
        for i in range(len(y_p)):
            if y_p[i] >= optm:
                y_p[i] = 1.0
            else:
                y_p[i] = 0.0
    else:
        for i in range(len(y_p)):
            if y_p[i] >= optm:
                y_p[i] = 0.0
            else:
                y_p[i] = 1.0
    return y_p

def optamize(yp,ye):
    corrected_yp = yp
    Tav = torch.tensor(0.0)
    Fav = torch.tensor(0.0)
    mean = torch.mean(yp)
    std = torch.std(yp)
    optm = torch.tensor(0.0)
    #Calculate Tav and Fav
    for i in range(len(yp)):
        if yp[i] > mean - 3*std and yp[i] < mean + 3*std:
            #average all expected True predictions
            if ye[i] > 0.5:
                
                if Tav == 0.0:
                    Tav = yp[i]
                else:
                    Tav = (Tav + yp[i]) / 2
                    
            #average all expected False predictions
            else:
                
                if Fav == 0.0:
                    Fav = yp[i]
                else:
                    Fav = (Fav + yp[i]) / 2         
    #compare Tav and Fav
    optm = (Tav+Fav)/2
    if(Tav<Fav):
        rev = True
        
    else:
        rev = False
    print(optm)
    corrected_yp = fixed_pred(yp,optm,rev)
    return optm, corrected_yp

#logic: for a threshold that gives the max accuracy, what is the false positive rate? ideally we want accuracy -> 1 and fpr -> 0
def evaluation_metrics(y_pred,freeze_y):
    optm, y_pred_optm = optamize(y_pred, freeze_y)
    avg = calc_acc(y_pred_optm, freeze_y)
    fpr = calc_FPR(y_pred_optm, freeze_y)
    return avg,fpr

def average_cross_val_metrics(x,y,model_type,criterion):
    accuracy= np.zeros((len(x[:,0,0])))
    FPR = np.zeros((len(x[:,0,0])))
    averaged_metrics = {}
    epochs = 10
    print(model_type)
    for i in range(len(x[:,0,0])):
        print("##cross val batch", i,"model type",model_type)
        loss_list = []
        #define the model
        model = make_model(model_type)
        #optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
        lr = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay= lr/100)
        freeze_x = x[i]
        freeze_y = y[i]
        train_x = x[np.arange(len(x))!=i]
        train_y = y[np.arange(len(y))!=i]
        #train model with everything in x and y - the ith index...
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            # Forward pass
            y_pred = model(torch.from_numpy(train_x).float())
            if(model_type == 'CNN'):
                loss = criterion(y_pred.squeeze(), torch.reshape(torch.from_numpy(train_y),[len(y_pred.squeeze())]).float())
            else:
                loss = criterion(y_pred.squeeze(), torch.from_numpy(train_y).squeeze().float())
            if epoch%2 == 0:
                print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
            loss_list.append(loss)
            # Backward pass
            loss.backward()
            optimizer.step()
        #validate the model with ith index and store the metrics...
        model.eval()
        print("y_pred")
        y_pred = model(torch.from_numpy(freeze_x).float())
        print(y_pred.shape,freeze_y.shape)
        print("calculating metrics")
        accuracy[i],FPR[i] = evaluation_metrics(y_pred,freeze_y)
        if(i==epochs-1):
            plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
            plt.plot(loss_list)
            plt.ylabel('MSE Loss')
            plt.xlabel('Epoch')
            title = 'Cross Validation Loss for '+model_type+', a = '+str(round(np.mean(accuracy),2))+', fpr = '+str(round(np.mean(FPR),2))+', alpha(LR) = '+str(lr)
            plt.title(title)
            plt.show()
        print("deleting model")
        #model.reset_parameters()
        del model
        print("done")
    #average the metrics and store in a dictionary...
    print("averaging metrics")
    averaged_metrics = {'accuracy': np.mean(accuracy), 'FPR':np.mean(FPR),}
    print("done")
    return averaged_metrics

#assuming x and y are of shapes [k,m/k,d] and [k,m/k,1] respectively.
def cross_validation(x=None,y=None,model_type='All'):
    results = {}
    if model_type == 'Linear':
        models = ['Linear']
    elif model_type == 'Perceptron':
        models = ['Linear']
    elif model_type == 'Shallow':
        models = ['Shallow']
    elif model_type == 'Deep':
        models = ['Deep']
    elif model_type == 'RNN':
        models = ['RNN']
    elif model_type == 'CNN':
        models = ['CNN']
    elif model_type == 'CNN_RNN':
        models = ['CNN_RNN']
    elif model_type == 'Pseudo_RNN':
        models = ['Pseudo_RNN']
    else:
        models = ['Linear','Perceptron','Shallow','Deep','Pseudo_RNN','RNN','CNN']
    for i in models: 
        criterion = torch.nn.MSELoss()      
        results[i] = average_cross_val_metrics(x,y,i,criterion)
        print("###########",results)
    return results

def make_model(model_type = 'Linear', print_m = False):
    if model_type == 'Linear':
        model = Linear_Regression(Ic_global, 1)
    elif model_type == 'Perceptron':
        model = Feedforward(Ic_global,1)
    elif model_type == 'Shallow':
        model = Shallow_Feedforward(Ic_global,4,1)
    elif model_type == 'Deep':
        model = Deep_Feedforward(Ic_global,100,80,40,20,10,1)
    elif model_type == 'RNN':
        model = RNN()
    elif model_type == 'CNN':
        model = CNN()
    elif model_type == 'CNN_RNN':
        model = CNN_RNN()
    else:
        model = Pseudo_RNN(Ic_global,1)
    if(print_m):
        print(model)
    return model
            
def train_model(x=None,y=None,model_type='Linear',epochs=20,lr= 0.001,decay=0.01,op='ADAM', momentum = 0,ls='BCE'):
    print('### Training ',model_type,'###')
    loss_list = []
    model = make_model(model_type)
    if ls == 'MSE':
        criterion = torch.nn.MSELoss()      
    else:
        criterion = torch.nn.BCELoss()
    if(op == 'SGD'):
        optimizer = torch.optim.SGD(model.parameters(), lr =lr, weight_decay= decay, momentum=momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr =lr, weight_decay= decay)
    model.train()
    for epoch in range(epochs): 
        optimizer.zero_grad()
        # Forward pass
        y_p = model(torch.from_numpy(x).float())
        if(model_type in ['CNN','CNN_RNN']):
            loss = criterion(y_p.squeeze(), torch.reshape(torch.from_numpy(y),[len(y_p.squeeze())]).float())
        else:
            loss = criterion(y_p.squeeze(), torch.from_numpy(y).squeeze().float())
        if epoch%int(epochs/10) == 0:
            print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
            print('## Estimated time left:', int(round(((epochs - epoch + 1)*6)/(10*60),0)),'min(s)', int(round(60 * (((epochs - epoch + 1)*6)/(10*60) - int(((epochs - epoch + 1)*6)/(10*60))),0)), 'second(s) ##')
        loss_list.append(loss)
        # Backward pass
        loss.backward()
        optimizer.step()
    return model, loss_list
    