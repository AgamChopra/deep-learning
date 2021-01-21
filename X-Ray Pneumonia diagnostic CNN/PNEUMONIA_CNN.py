import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import image
from matplotlib import pyplot

def load_set_index():
    
    df = pd.read_excel (r'chest_xray\dataset.xlsx')
    index = df.dropna().values
    
    #outputs 9 values(train(n,b,v),test(n,b,v),val(n,b,v))
    return (int(index[0,1]),int(index[1,1]),int(index[2,1]),int(index[0,2]),int(index[1,2]),int(index[2,2]),int(index[0,3]),int(index[1,3]),int(index[2,3]))   
	
#code LI0 : Incorrect set name.
#code LI1 : Incorrect scan type.

def load_image_as_array(set_name,scan_type,i):
    
    image = np.zeros((400,400))
    
    if set_name == "val":
        
            if scan_type == 0:    
                    image = np.asarray(Image.open(r'chest_xray\chest_xray\val\NORMAL\img (%d).jpeg' % (i+1)).convert("L"))             
            
            elif scan_type == 1:   
                    image = np.asarray(Image.open(r'chest_xray\chest_xray\val\PNEUMONIA\BACTERIA\img (%d).jpeg' % (i+1)).convert("L"))             
            
            elif scan_type == 2:     
                    image = np.asarray(Image.open(r'chest_xray\chest_xray\val\PNEUMONIA\VIRUS\img (%d).jpeg' % (i+1)).convert("L"))               
            
            else:
                print("Error code LI1... Unable to load image, incorrect scan type.")            
    
    elif set_name == "test":
        
            if scan_type == 0:
                    image = np.asarray(Image.open(r'chest_xray\chest_xray\test\NORMAL\img (%d).jpeg' % (i+1)).convert("L"))             
            
            elif scan_type == 1:
                    image = np.asarray(Image.open(r'chest_xray\chest_xray\test\PNEUMONIA\BACTERIA\img (%d).jpeg' % (i+1)).convert("L"))             
            
            elif scan_type == 2:
                    image = np.asarray(Image.open(r'chest_xray\chest_xray\test\PNEUMONIA\VIRUS\img (%d).jpeg' % (i+1)).convert("L"))     
            
            else:
                print("Error code LI1... Unable to load image, incorrect scan type.")                     
    
    elif set_name == "train":
        
            if scan_type == 0:
                    image = np.asarray(Image.open(r'chest_xray\chest_xray\train\NORMAL\img (%d).jpeg' % (i+1)).convert("L"))
            
            elif scan_type == 1:
                    image = np.asarray(Image.open(r'chest_xray\chest_xray\train\PNEUMONIA\BACTERIA\img (%d).jpeg' % (i+1)).convert("L"))             
            
            elif scan_type == 2:     
                    image = np.asarray(Image.open(r'chest_xray\chest_xray\train\PNEUMONIA\VIRUS\img (%d).jpeg' % (i+1)).convert("L"))        
            
            else:
                print("Error code LI1... Unable to load image, incorrect scan type.")                      
    
    else:
        print("Error code LI0... Unable to load image, incorrect set name.")
         
    return(image)

def load_norm_img(set_name,scan_type,i):
    return(load_image_as_array(set_name,scan_type,i)/255)

def populate_index(y):
    index = y[:,0].size
    for i in range(0,index):
        y[i,0] = i
    return y

def make_tags(set_name):
    index = load_set_index()
    if set_name == "train":
        y_train_n = populate_index(np.zeros((index[0],2),dtype=int))
        y_train_b = populate_index(np.ones((index[1],2),dtype=int))
        y_train_v = populate_index(2*np.ones((index[2],2),dtype=int))
        y_set = np.concatenate((y_train_n,y_train_b,y_train_v),axis=0) 
    elif set_name == "test":
        y_test_n = populate_index(np.zeros((index[3],2),dtype=int))
        y_test_b = populate_index(np.ones((index[4],2),dtype=int))
        y_test_v = populate_index(2*np.ones((index[5],2),dtype=int))
        y_set = np.concatenate((y_test_n,y_test_b,y_test_v),axis=0) 
    elif set_name == "val":
        y_val_n = populate_index(np.zeros((index[6],2),dtype=int))
        y_val_b = populate_index(np.ones((index[7],2),dtype=int))
        y_val_v = populate_index(2*np.ones((index[8],2),dtype=int))
        y_set = np.concatenate((y_val_n,y_val_b,y_val_v),axis=0)
    else:
        print("Error in make_tag(): Incorrect set name '%s'\nmake_tag() returned 'None'" % (setname))
        return None
    return y_set

def make_xy_sets(set_name):
    y_temp = make_tags(set_name)
    y_set = np.zeros((y_temp[:,0].size))
    x_set = np.zeros((y_temp[:,0].size,400,400))
    
    np.random.shuffle(y_temp)
    
    for i in range(y_temp[:,0].size):
        y_set[i] = y_temp[i,1]
        x_set[i,:,:] = load_norm_img(set_name,y_temp[i,1],0)   
    return x_set,y_set

#print(make_xy_sets("val"))

"""

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

def load_data_sets():
    # setup
    index = load_image_data()
    
    x_train = np.zeros((400,400,index[0]+index[1]+index[2]))
    y_train = np.zeros((index[0]+index[1]+index[2]))
    
    x_test = np.zeros((400,400,index[3]+index[4]+index[5]))
    y_test = np.zeros((index[3]+index[4]+index[5]))
    
    x_val = np.zeros((400,400,index[6]+index[7]+index[8]))
    y_val = np.zeros((index[6]+index[7]+index[8]))
    
    n_v,b_v,v_v,n_ts,b_ts,v_ts,n_tr,b_tr,v_tr = load_image_set(index[0],index[1],index[2],index[3],index[4],index[5],index[6],index[7],index[8])
    print(n_v.shape)
    return x_train,y_train,x_test,y_test,x_val,y_val
  
x_train,y_train,x_test,y_test,x_val,y_val = load_data_sets()
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape,x_val.shape,y_val.shape)
print(x_val,y_val)
"""