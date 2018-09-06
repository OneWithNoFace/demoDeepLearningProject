import os
from PIL import Image
import numpy as np 
from keras.utils import np_utils
from models.cnn1.lenet import CNN1
from keras.optimizers import SGD

def get_label(name):
  if "agricultural" in name:
    return 0
  if "airplane" in name:
    return 1
  if "beach" in name:
    return 2
  if "chaparral" in name:
    return 3
  if "denseresidential" in name:
    return 4
  if "freeway" in name:
    return 5
  if "overpass" in name:
    return 6
  if "parkinglot" in name:
    return 7
  if "storagetank" in name:
    return 8
  return -1

def get_all_images_as_nparray(folder_path):
  data = []
  labels = []
  for fname in os.listdir(folder_path):
    im = Image.open(os.path.join(folder_path,fname))
    data.append(np.array(im))
    labels.append(get_label(fname))
  return np.asarray(data),np.asarray(labels)

(test_data,test_labels) = get_all_images_as_nparray('./demoDeepLearningProject/data/test/')
(train_data,train_labels) = get_all_images_as_nparray('./demoDeepLearningProject/data/train/')

test_labels = test_labels[:, None]
train_labels = train_labels[:, None]

print(test_data.shape,test_labels.shape)
print(train_data.shape,train_labels.shape)

test_labels = np_utils.to_categorical(test_labels,9)
train_labels = np_utils.to_categorical(train_labels,9)

test_data = test_data.astype('float32')/255.0
train_data = train_data.astype('float32')/255.0

model = CNN1.build(num_channels=3,img_rows=256,img_cols=256,num_classes=9,activation='relu')

model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.01), metrics=["accuracy"])

model.fit(train_data,train_labels,batch_size=512,epochs=1)