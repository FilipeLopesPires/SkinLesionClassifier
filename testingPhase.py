from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.layers import LeakyReLU

import matplotlib.pyplot as plt

# dimensions of our images.
img_width, img_height = 200,200

validation_data_dir = 'test/normalizedTest/testMelanoma'
nb_train_samples = 2000
nb_validation_samples = 600
epochs = 50
batch_size = 8

kernel_size=3

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)



model = Sequential()
model.add(Conv2D(64, kernel_size, input_shape=input_shape))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

#
model.add(Conv2D(64, kernel_size, input_shape=input_shape))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, kernel_size))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(64))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.1))

model.add(Dense(64))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.1))


#model.add(Dropout(0.3))
    

model.add(Dense(64))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.1))


model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



model.load_weights('normalized_200_50_8_64_3-arch3convpool3denseKERADROP.h5')




model2 = Sequential()
model2.add(Conv2D(64, kernel_size, input_shape=input_shape))
#model.add(Activation('relu'))
model2.add(LeakyReLU(alpha=0.1))
model2.add(MaxPooling2D(pool_size=(2, 2)))

#
model2.add(Conv2D(64, kernel_size, input_shape=input_shape))
#model.add(Activation('relu'))
model2.add(LeakyReLU(alpha=0.1))
model2.add(MaxPooling2D(pool_size=(2, 2)))


model2.add(Conv2D(64, kernel_size))
#model.add(Activation('relu'))
model2.add(LeakyReLU(alpha=0.1))
model2.add(MaxPooling2D(pool_size=(2, 2)))


model2.add(Flatten())
model2.add(Dense(64))
#model.add(Activation('relu'))
model2.add(LeakyReLU(alpha=0.1))

#model2.add(Dropout(0.5))

model2.add(Dense(64))
#model.add(Activation('relu'))
model2.add(LeakyReLU(alpha=0.1))



model2.add(Dense(64))
#model.add(Activation('relu'))
model2.add(LeakyReLU(alpha=0.1))


model2.add(Dense(1))
model2.add(Activation('sigmoid'))

model2.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



model2.load_weights('normalized_200_50_8_64_3-arch3convpool3dense.h5')


test_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen2 = ImageDataGenerator(rescale=1. / 255)

mela = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


kera = test_datagen2.flow_from_directory(
    "test/normalizedTest/testKeratosis",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')



#accuracies

acckera = model.evaluate_generator(
    kera,
    steps=nb_validation_samples // batch_size)


accmela = model2.evaluate_generator(
    mela,
    steps=nb_validation_samples // batch_size)

print(acckera, accmela)




kerapred = model.predict_generator(
    kera,
    steps=nb_validation_samples // batch_size)

melapred = model2.predict_generator(
    mela,
    steps=nb_validation_samples // batch_size)


kerapred = kerapred.ravel()
melapred = melapred.ravel()

ymela=[]
ykera=[]

f=open('test/ISIC-2017_Test_v2_Part3_GroundTruth.csv', 'r')
f.readline()

for line in f.readlines():
    file=line.split(",")[0]
    melanoma=line.split(",")[1]
    keratosis=line.split(",")[2]

        
    if float(melanoma)==1:
        ymela.append(1)
    else:
        ymela.append(0)
    
    if float(keratosis)==1:
        ykera.append(1)
    else:
        ykera.append(0)
    
f.close()

totalrightmela=0
totalrightkera=0

for i in range(len(kerapred)):
    k=1 if kerapred[i] > 0.5 else 0
    m=1 if melapred[i] > 0.5 else 0
    if k==ykera[i]:
        totalrightkera+=1
    if m==ymela[i]:
        totalrightmela+=1


y_total=[]
for i in range(len(ykera)):
    if ymela[i]==1:
        y_total.append(0)
    elif ykera[i]==1:
        y_total.append(1)
    else:
        y_total.append(3)


pred_total=[]
for i in range(len(kerapred)):
    if melapred[i]==1 and kerapred[i]==1:
        pred_total.append(1)
    elif melapred[i]==1:
        pred_total.append(0)
    elif kerapred[i]==1:
        pred_total.append(1)
    else:
        pred_total.append(3)

print(len(pred_total))
aux=0
for i in range(len(pred_total)):
    if pred_total[i]==y_total[i]:
        aux+=1
print(aux)