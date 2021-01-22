import pandas as pd
import numpy as np
from PIL import Image
#from matplotlib import image
from matplotlib import pyplot as plt
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
import keras.backend as K
K.set_image_data_format('channels_last')
import tensorflow as tf

#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
#tf.test.is_built_with_gpu_support()

def load_set_index():
    
    df = pd.read_excel (r'chest_xray\dataset.xlsx')
    index = df.dropna().values
    
    #outputs 9 values(train(n,b,v),test(n,b,v),val(n,b,v))
    return (int(index[0,1]),int(index[1,1]),int(index[2,1]),int(index[0,2]),int(index[1,2]),int(index[2,2]),int(index[0,3]),int(index[1,3]),int(index[2,3]))   
	
#code LI0 : Incorrect set name.
#code LI1 : Incorrect scan type.

def load_image_as_array(set_name,scan_type,i):
    
    imagee = np.zeros((400,400))
    
    if set_name == "val":
        
            if scan_type == 0:    
                    imagee = np.asarray(Image.open(r'chest_xray\chest_xray\val\NORMAL\img (%d).jpeg' % (i+1)).convert("L"))             
            
            elif scan_type == 1:   
                    imagee = np.asarray(Image.open(r'chest_xray\chest_xray\val\PNEUMONIA\BACTERIA\img (%d).jpeg' % (i+1)).convert("L"))             
            
            elif scan_type == 2:     
                    imagee = np.asarray(Image.open(r'chest_xray\chest_xray\val\PNEUMONIA\VIRUS\img (%d).jpeg' % (i+1)).convert("L"))               
            
            else:
                print("Error code LI1... Unable to load image, incorrect scan type.")            
    
    elif set_name == "test":
        
            if scan_type == 0:
                    imagee = np.asarray(Image.open(r'chest_xray\chest_xray\test\NORMAL\img (%d).jpeg' % (i+1)).convert("L"))             
            
            elif scan_type == 1:
                    imagee = np.asarray(Image.open(r'chest_xray\chest_xray\test\PNEUMONIA\BACTERIA\img (%d).jpeg' % (i+1)).convert("L"))             
            
            elif scan_type == 2:
                    imagee = np.asarray(Image.open(r'chest_xray\chest_xray\test\PNEUMONIA\VIRUS\img (%d).jpeg' % (i+1)).convert("L"))     
            
            else:
                print("Error code LI1... Unable to load image, incorrect scan type.")                     
    
    elif set_name == "train":
        
            if scan_type == 0:
                    imagee = np.asarray(Image.open(r'chest_xray\chest_xray\train\NORMAL\img (%d).jpeg' % (i+1)).convert("L"))
            
            elif scan_type == 1:
                    imagee = np.asarray(Image.open(r'chest_xray\chest_xray\train\PNEUMONIA\BACTERIA\img (%d).jpeg' % (i+1)).convert("L"))             
            
            elif scan_type == 2:     
                    imagee = np.asarray(Image.open(r'chest_xray\chest_xray\train\PNEUMONIA\VIRUS\img (%d).jpeg' % (i+1)).convert("L"))        
            
            else:
                print("Error code LI1... Unable to load image, incorrect scan type.")                      
    
    else:
        print("Error code LI0... Unable to load image, incorrect set name.")
         
    return(imagee)

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
        print("Error in make_tag(): Incorrect set name '%s'\nmake_tag() returned 'None'" % (set_name))
        return None
    return y_set

def make_xy_sets(set_name):
    y_temp = make_tags(set_name)
    y_set = np.zeros((y_temp[:,0].size))
    x_set = np.zeros((y_temp[:,0].size,400,400,1))
    
    np.random.shuffle(y_temp)
    
    for i in range(y_temp[:,0].size):
        y_set[i] = y_temp[i,1]
        x_set[i,:,:,0] = load_norm_img(set_name,y_temp[i,1],0)   
    return x_set,y_set

#print(make_xy_sets("val"))
'''
X_train,Y_train = make_xy_sets("train")
print(X_train.shape, X_train.shape[1:4])
plt.imshow(X_train[0,:,:,0],cmap = "bone")

def TestCNN(input_shape,classes = 3):
    
    X_input = Input(input_shape)
        
    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)
        
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
        
    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)
        
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)
        
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='testmodel')
    
    ### END CODE HERE ###
    
    return model
'''

def identity_block(X, f, filters, stage, block):
   
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X
    
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X

    X = Conv2D(F1, (1, 1), strides = (s,s),padding = 'valid', name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(F2, (f, f), strides = (1,1),padding = 'same', name = conv_name_base + '2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(F3, (1, 1), strides = (1,1),padding = 'valid', name = conv_name_base + '2c')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s),padding = 'valid', name = conv_name_base + '1')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X

def ResNet50(input_shape = (64, 64, 3), classes = 6):
    
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)
    
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)
    
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)
    
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

X_train,Y_train = make_xy_sets("train")
plt.imshow(X_train[0,:,:,0],cmap = "bone")
Y_train = tf.keras.utils.to_categorical(Y_train, 3)
print(Y_train.shape)
X_test,Y_test = make_xy_sets("test")
Y_test = tf.keras.utils.to_categorical(Y_test, 3)
print(Y_test.shape)
X_v,Y_v = make_xy_sets("val")
Y_v = tf.keras.utils.to_categorical(Y_v, 3)
print(Y_v.shape)

testmodel = ResNet50(X_train.shape[1:4],classes=3)
testmodel.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
testmodel.summary()

testmodel.fit(x = X_train, y = Y_train, epochs = 5, batch_size = 24) #test model

preds = testmodel.evaluate(x = X_test, y = Y_test)
print ("\nLoss = " + str(preds[0]) + "\nTest Accuracy = " + str(preds[1]))

predsv = testmodel.evaluate(x = X_v, y = Y_v)
print ("\nLoss = " + str(predsv[0]) + "\nTest Accuracy = " + str(predsv[1]))