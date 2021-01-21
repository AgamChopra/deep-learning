# Image to NumPy array test code by Agamdeep S. Chopra.
import pandas as pd
from numpy import asarray as asr
from PIL import Image
from matplotlib import image
from matplotlib import pyplot

df = pd.read_excel (r'XYZ\chest_xray\dataset.xlsx')
index = df.dropna().values

train_normal_index = int(index[0,1])
train_bacteria_index = int(index[1,1])
train_viral_index = int(index[2,1])

test_normal_index = int(index[0,2])
test_bacteria_index = int(index[1,2])
test_viral_index = int(index[2,2])

val_normal_index = int(index[0,3])
val_bacteria_index = int(index[1,3])
val_viral_index = int(index[2,3])

print("Dataset image index table:\n",index)
print("\nTraining data ->",train_normal_index,train_bacteria_index,train_viral_index)
print("\nTesting data ->",test_normal_index,test_bacteria_index,test_viral_index)
print("\nValidation data ->",val_normal_index,val_bacteria_index,val_viral_index)

# Pillow(PIL) Approach

test_image_PIL = Image.open(r'chest_xray\chest_xray\val\NORMAL\img (1).jpeg')

print(test_image_PIL)
print(test_image_PIL.format)
print(test_image_PIL.size)
print(test_image_PIL.mode)# L means 8-bit pixels, black and white

test_image_PIL.show()

test_data_PIL_numpy = asr(test_image_PIL)
print(type(test_data_PIL_numpy))
print(test_data_PIL_numpy.shape)
print(test_data_PIL_numpy)

# MATPLOTLIB Approach

test_image_MPL = image.imread(r'chest_xray\chest_xray\val\NORMAL\img (1).jpeg')

print(test_image_MPL.dtype)
print(test_image_MPL.shape)

pyplot.imshow(test_image_MPL,cmap='gray')
pyplot.show()

test_data_MPL_numpy = asr(test_image_MPL)
print(type(test_data_MPL_numpy))
print(test_data_MPL_numpy.shape)
print(test_data_MPL_numpy)

for i in range(1,val_normal_index+1):
    test_image_MPL = image.imread(r'chest_xray\chest_xray\val\NORMAL\img (%d).jpeg' % (i))
    print("\nImage %d:" % (i))
    #pyplot.imshow(test_image_MPL,cmap='gray')
    #pyplot.show()
    pyplot.imshow(test_image_MPL,cmap='bone')
    pyplot.show()