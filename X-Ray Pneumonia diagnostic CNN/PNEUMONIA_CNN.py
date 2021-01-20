import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import image
from matplotlib import pyplot

def load_image_data():
    
    df = pd.read_excel (r'chest_xray\dataset.xlsx')
    index = df.dropna().values
    
    #outputs 9 values(train(n,b,v),test(n,b,v),val(n,b,v))
    return (int(index[0,1]),int(index[1,1]),int(index[2,1]),int(index[0,2]),int(index[1,2]),int(index[2,2]),int(index[0,3]),int(index[1,3]),int(index[2,3]))   

def load_image_set(train_n,train_b,train_v,test_n,test_b,test_v,val_n,val_b,val_v):
    val_n_array = np.zeros((400,400,val_n))
    val_b_array = np.zeros((400,400,val_b))
    val_v_array = np.zeros((400,400,val_v))
    
    test_n_array = np.zeros((400,400,test_n))
    test_b_array = np.zeros((400,400,test_b))
    test_v_array = np.zeros((400,400,test_v))
    
    train_n_array = np.zeros((400,400,train_n))
    train_b_array = np.zeros((400,400,train_b))
    train_v_array = np.zeros((400,400,train_v))


    for i in range(val_n):
        val_n_array[:,:,i] = np.asarray(Image.open(r'chest_xray\chest_xray\val\NORMAL\img (%d).jpeg' % (i+1)).convert("L"))
    for i in range(val_b):
        val_b_array[:,:,i] = np.asarray(Image.open(r'chest_xray\chest_xray\val\PNEUMONIA\BACTERIA\img (%d).jpeg' % (i+1)).convert("L"))    
    for i in range(val_v):
        val_v_array[:,:,i] = np.asarray(Image.open(r'chest_xray\chest_xray\val\PNEUMONIA\VIRUS\img (%d).jpeg' % (i+1)).convert("L"))
    for i in range(test_n):
        test_n_array[:,:,i] = np.asarray(Image.open(r'chest_xray\chest_xray\test\NORMAL\img (%d).jpeg' % (i+1)).convert("L"))
    for i in range(test_b):
        test_b_array[:,:,i] = np.asarray(Image.open(r'chest_xray\chest_xray\test\PNEUMONIA\BACTERIA\img (%d).jpeg' % (i+1)).convert("L"))    
    for i in range(test_v):
        test_v_array[:,:,i] = np.asarray(Image.open(r'chest_xray\chest_xray\test\PNEUMONIA\VIRUS\img (%d).jpeg' % (i+1)).convert("L"))
    for i in range(train_n):
        train_n_array[:,:,i] = np.asarray(Image.open(r'chest_xray\chest_xray\train\NORMAL\img (%d).jpeg' % (i+1)).convert("L"))
    for i in range(train_b):
        train_b_array[:,:,i] = np.asarray(Image.open(r'chest_xray\chest_xray\train\PNEUMONIA\BACTERIA\img (%d).jpeg' % (i+1)).convert("L"))    
    for i in range(train_v):
        train_v_array[:,:,i] = np.asarray(Image.open(r'chest_xray\chest_xray\train\PNEUMONIA\VIRUS\img (%d).jpeg' % (i+1)).convert("L"))
              
    return  (val_n_array,val_b_array,val_v_array,test_n_array,test_b_array,test_v_array,train_n_array,train_b_array,train_v_array)