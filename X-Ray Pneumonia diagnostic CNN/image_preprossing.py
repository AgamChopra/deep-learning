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

#Analyzing the image dataset

x_small = 0
y_small = 0
counter = 0

for i in range(1,val_normal_index+1):
    test_image_PIL = Image.open(r'chest_xray\chest_xray\val\NORMAL\img (%d).jpeg' % (i)).convert("L")
    test_image_MPL = asr(test_image_PIL)
    x_temp, y_temp = test_image_MPL.shape
    if x_small == 0 or x_small > x_temp:
        x_small = x_temp
    if y_small == 0 or y_small > y_temp:
        y_small = y_temp
    counter+=1
for i in range(1,val_bacteria_index+1):
    test_image_PIL = Image.open(r'chest_xray\chest_xray\val\PNEUMONIA\BACTERIA\img (%d).jpeg' % (i)).convert("L")
    test_image_MPL = asr(test_image_PIL)
    x_temp, y_temp = test_image_MPL.shape
    if x_small > x_temp:
        x_small = x_temp
    if y_small > y_temp:
        y_small = y_temp
    counter+=1
for i in range(1,val_viral_index+1):
    test_image_PIL = Image.open(r'chest_xray\chest_xray\val\PNEUMONIA\VIRUS\img (%d).jpeg' % (i)).convert("L")
    test_image_MPL = asr(test_image_PIL)
    x_temp, y_temp = test_image_MPL.shape
    if x_small > x_temp:
        x_small = x_temp
    if y_small > y_temp:
        y_small = y_temp
    counter+=1
    
print(x_small, y_small)

for i in range(1,test_normal_index+1):
    test_image_PIL = Image.open(r'chest_xray\chest_xray\test\NORMAL\img (%d).jpeg' % (i)).convert("L")
    test_image_MPL = asr(test_image_PIL)
    x_temp, y_temp= test_image_MPL.shape
    if x_small == 0 or x_small > x_temp:
        x_small = x_temp
    if y_small == 0 or y_small > y_temp:
        y_small = y_temp
    counter+=1
for i in range(1,test_bacteria_index+1):
    test_image_PIL = Image.open(r'chest_xray\chest_xray\test\PNEUMONIA\BACTERIA\img (%d).jpeg' % (i)).convert("L")
    test_image_MPL = asr(test_image_PIL)
    x_temp, y_temp= test_image_MPL.shape
    if x_small > x_temp:
        x_small = x_temp
    if y_small > y_temp:
        y_small = y_temp
    counter+=1
for i in range(1,test_viral_index+1):
    test_image_PIL = Image.open(r'chest_xray\chest_xray\test\PNEUMONIA\VIRUS\img (%d).jpeg' % (i)).convert("L")
    test_image_MPL = asr(test_image_PIL)
    x_temp, y_temp= test_image_MPL.shape
    if x_small > x_temp:
        x_small = x_temp
    if y_small > y_temp:
        y_small = y_temp
    counter+=1

print(x_small, y_small) 

for i in range(1,train_normal_index+1):
    test_image_PIL = Image.open(r'chest_xray\chest_xray\train\NORMAL\img (%d).jpeg' % (i)).convert("L")
    test_image_MPL = asr(test_image_PIL)
    x_temp, y_temp= test_image_MPL.shape
    if x_small == 0 or x_small > x_temp:
        x_small = x_temp
    if y_small == 0 or y_small > y_temp:
        y_small = y_temp
    counter+=1
print(x_small, y_small)
for i in range(1,train_bacteria_index+1):
    test_image_PIL = Image.open(r'chest_xray\chest_xray\train\PNEUMONIA\BACTERIA\img (%d).jpeg' % (i)).convert("L")
    test_image_MPL = asr(test_image_PIL)
    x_temp, y_temp= test_image_MPL.shape
    if x_small > x_temp:
        x_small = x_temp
    if y_small > y_temp:
        y_small = y_temp
    counter+=1
print(x_small, y_small)
for i in range(1,train_viral_index+1):
    test_image_PIL = Image.open(r'chest_xray\chest_xray\train\PNEUMONIA\VIRUS\img (%d).jpeg' % (i)).convert("L")
    test_image_MPL = asr(test_image_PIL)
    x_temp, y_temp= test_image_MPL.shape
    if x_small > x_temp:
        x_small = x_temp
    if y_small > y_temp:
        y_small = y_temp
    counter+=1
    
print("Size of the smallest image is: x = %d and y = %d" %(x_small, y_small))
print("Total images:",counter)

#image resizing

print("Processing") 
for i in range(1,val_normal_index+1):
    image = Image.open(r'chest_xray\chest_xray\val\NORMAL\img (%d).jpeg' % (i)).convert("L")
    new_image = image.resize((4000, 4000))
    new_image = new_image.crop((500,500,3500,3500))
    new_image = new_image.resize((400, 400))
    new_image.save(r'chest_xray\chest_xray\val\NORMAL\img (%d).jpeg' % (i))
print("Done")
print("Processing") 
for i in range(1,val_bacteria_index+1):
    image = Image.open(r'chest_xray\chest_xray\val\PNEUMONIA\BACTERIA\img (%d).jpeg' % (i)).convert("L")
    new_image = image.resize((4000, 4000))
    new_image = new_image.crop((500,500,3500,3500))
    new_image = new_image.resize((400, 400))
    new_image.save(r'chest_xray\chest_xray\val\PNEUMONIA\BACTERIA\img (%d).jpeg' % (i))
print("Done")
print("Processing")     
for i in range(1,val_viral_index+1):
    image = Image.open(r'chest_xray\chest_xray\val\PNEUMONIA\VIRUS\img (%d).jpeg' % (i)).convert("L")
    new_image = image.resize((4000, 4000))
    new_image = new_image.crop((500,500,3500,3500))
    new_image = new_image.resize((400, 400))
    new_image.save(r'chest_xray\chest_xray\val\PNEUMONIA\VIRUS\img (%d).jpeg' % (i))
print("Done")
print("Processing") 
for i in range(1,test_normal_index+1):
    image = Image.open(r'chest_xray\chest_xray\test\NORMAL\img (%d).jpeg' % (i)).convert("L")
    new_image = image.resize((4000, 4000))
    new_image = new_image.crop((500,500,3500,3500))
    new_image = new_image.resize((400, 400))
    new_image.save(r'chest_xray\chest_xray\test\NORMAL\img (%d).jpeg' % (i))
print("Done")
print("Processing") 
for i in range(1,test_bacteria_index+1):
    image = Image.open(r'chest_xray\chest_xray\test\PNEUMONIA\BACTERIA\img (%d).jpeg' % (i)).convert("L")
    new_image = image.resize((4000, 4000))
    new_image = new_image.crop((500,500,3500,3500))
    new_image = new_image.resize((400, 400))
    new_image.save(r'chest_xray\chest_xray\test\PNEUMONIA\BACTERIA\img (%d).jpeg' % (i))
print("Done")
print("Processing")     
for i in range(1,test_viral_index+1):
    image = Image.open(r'chest_xray\chest_xray\test\PNEUMONIA\VIRUS\img (%d).jpeg' % (i)).convert("L")
    new_image = image.resize((4000, 4000))
    new_image = new_image.crop((500,500,3500,3500))
    new_image = new_image.resize((400, 400))
    new_image.save(r'chest_xray\chest_xray\test\PNEUMONIA\VIRUS\img (%d).jpeg' % (i))
print("Done")
print("Processing") 
for i in range(1,train_normal_index+1):
    image = Image.open(r'chest_xray\chest_xray\train\NORMAL\img (%d).jpeg' % (i)).convert("L")
    new_image = image.resize((4000, 4000))
    new_image = new_image.crop((500,500,3500,3500))
    new_image = new_image.resize((400, 400))
    new_image.save(r'chest_xray\chest_xray\train\NORMAL\img (%d).jpeg' % (i))
print("Done")
print("Processing") 
for i in range(1,train_bacteria_index+1):
    image = Image.open(r'chest_xray\chest_xray\train\PNEUMONIA\BACTERIA\img (%d).jpeg' % (i)).convert("L")
    new_image = image.resize((4000, 4000))
    new_image = new_image.crop((500,500,3500,3500))
    new_image = new_image.resize((400, 400))
    new_image.save(r'chest_xray\chest_xray\train\PNEUMONIA\BACTERIA\img (%d).jpeg' % (i))
print("Done")
print("Processing")     
for i in range(1,train_viral_index+1):
    image = Image.open(r'chest_xray\chest_xray\train\PNEUMONIA\VIRUS\img (%d).jpeg' % (i)).convert("L")
    new_image = image.resize((4000, 4000))
    new_image = new_image.crop((500,500,3500,3500))
    new_image = new_image.resize((400, 400))
    new_image.save(r'chest_xray\chest_xray\train\PNEUMONIA\VIRUS\img (%d).jpeg' % (i))
print("Done")
print("Complete!") 

from matplotlib.pyplot import figure
gcolor = "bone"
i = 13
test_image_MPL = Image.open(r'chest_xray\chest_xray\test\NORMAL\img (%d).jpeg' % (i)).convert("L")
print("\nImage Normal")
figure(num=None, figsize=(3,3), dpi=180, facecolor='w', edgecolor='k')
pyplot.imshow(test_image_MPL,cmap=gcolor)
pyplot.show()
test_image_MPL = Image.open(r'chest_xray\chest_xray\test\PNEUMONIA\BACTERIA\img (%d).jpeg' % (i)).convert("L")
print("\nImage Bacterial:")
figure(num=None, figsize=(3,3), dpi=180, facecolor='w', edgecolor='k')
pyplot.imshow(test_image_MPL,cmap=gcolor)
pyplot.show()
test_image_MPL = Image.open(r'chest_xray\chest_xray\test\PNEUMONIA\VIRUS\img (%d).jpeg' % (i)).convert("L")
print("\nImage Viral:")
figure(num=None, figsize=(3,3), dpi=180, facecolor='w', edgecolor='k')
pyplot.imshow(test_image_MPL,cmap=gcolor)
pyplot.show()