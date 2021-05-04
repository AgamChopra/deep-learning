# main.py
# Author: Agamdeep S. Chopra, bit.ly/AgamChopra
# Last Updated: 05/01/2021

import sys 
import datasets as dp
from matplotlib import pyplot as plt 
import matplotlib.colors as mcolors
import models as mod
import json
from playsound import playsound

def main(PATH, sd):
    mod.Ic_global = 196
    ic = mod.Ic_global + 1 - 1
    train_x, train_y, test_x, test_y, val_x, val_y = dp.processed_data(sd = 4, reduce_feature = "pca", pca_k = ic)
    print(train_x.shape,train_y.shape,test_x.shape,test_y.shape, val_x.shape, val_y.shape)
    
    x,y = dp.k_split(train_x,train_y,10)
    print(x.shape,y.shape)
    results = []
    for i in range(1):
        results.append(mod.cross_validation(x, y,model_type='All'))
        print(results)
        print(len(results))
    #with open('cross-val-results', 'w') as fout:
        #json.dump(results, fout)
    # ^Uncomment if cross val results r needed...
    
    print("##Cross-Validation Results##")   
    m=['Linear','Perceptron','Shallow','Deep','Pseudo_RNN','RNN','CNN']
    for i in m:
        print('_________',i,'\naccuracy:',round(results[0][i]['accuracy'],3),', fpr:',round(results[0][i]['FPR'],3),'\n____________________________')
    
    mod.Ic_global = ic + 1 - 1
    lr = 0.0001
    #model = mod.train_model(x=train_x,y=train_y,model_type='CNN',epochs=100000,lr= lr,decay=lr/100, op='ADAM', ls='BCE')
    #model = mod.train_model(x=train_x,y=train_y,model_type='CNN',epochs=100000,lr= lr,decay=lr/100, op='ADAM', ls='MSE')
    model = mod.train_model(x=train_x,y=train_y,model_type='CNN',epochs=1000,lr= lr,decay=lr/100, op='ADAM', ls='MSE')
    # ^I lowered the epochs to 1000 for quicker training. The results should not be that different.
    
    model[0].eval()
    print("y_pred")
    yp = model[0](mod.torch.from_numpy(test_x).float())
    playsound(sd)
    
    plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(model[1])
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.title('Training Loss MSE')
    plt.show()
    
    plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    [plt.plot(yp.detach().numpy()[i],i/test_y.size,'.',c=(mcolors.CSS4_COLORS["teal"] if test_y[i] > 0.5 else mcolors.CSS4_COLORS["firebrick"])) for i in range(len(test_y))]
    plt.ylabel('normalized example #')
    plt.xlabel('prediction Value')
    plt.title('Prediction Clusters MSE')
    plt.show()
    
    yp = model[0](mod.torch.from_numpy(test_x).float())
    ye = test_y
    optm = 0.7
    rev = False
    plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    [plt.plot(yp.detach().numpy()[i],i/test_y.size,'.',c=(mcolors.CSS4_COLORS["teal"] if ye[i] > 0.5 else mcolors.CSS4_COLORS["firebrick"])) for i in range(len(yp))] 
    [plt.plot(optm,0+i/ye.size,'|',color='red') for i in range(len(yp))]
    plt.ylabel('example #')
    plt.xlabel('Prediction Value, Threshold {0}'.format(optm))
    plt.title('Test Accuracy {0}, FPR {1}'.format(round(mod.calc_acc(mod.fixed_pred(yp+1-1,optm,rev),ye),2),round(mod.calc_FPR(mod.fixed_pred(yp+1-1,optm,rev),ye),2)))
    plt.show()
    
    yp = model[0](mod.torch.from_numpy(val_x).float())
    ye = val_y
    rev = False
    plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    [plt.plot(yp.detach().numpy()[i],i/ye.size,'.',c=(mcolors.CSS4_COLORS["teal"] if ye[i] > 0.5 else mcolors.CSS4_COLORS["firebrick"])) for i in range(len(yp))] 
    [plt.plot(optm,0+i/ye.size,'|',color='red') for i in range(len(yp))]
    plt.ylabel('example #')
    plt.xlabel('Prediction Value, Threshold {0}'.format(optm))
    plt.title('Validation Accuracy {0}, FPR {1}'.format(round(mod.calc_acc(mod.fixed_pred(yp+1-1,optm,rev),ye),2),round(mod.calc_FPR(mod.fixed_pred(yp+1-1,optm,rev),ye),2)))
    plt.show()
    
    path = PATH + '\cnn.txt'
    mod.torch.save(model[0],path)
    return 'Complete!'

if __name__ == "__main__":
    val = input("Enter PATH: ")
    try:
        PATH = 'r' + val
        sys.path.append(PATH)
        dp.PATH = PATH
        sd = PATH + '\iphone-alarm-radar.mp3'
        task = main(PATH,sd)
        print(task)
    except:
        _ = input("Error!, Incorrect PATH. If this error persists, please contact the Author.")