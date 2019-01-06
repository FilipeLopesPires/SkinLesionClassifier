from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.layers import LeakyReLU


# dimensions of our images.
#150/200/250
img_width, img_height = 200, 200


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


train_data_dir = 'train/normalizedTrain/trainKeratosis'
validation_data_dir = 'validation/normalizedValidation/validationKeratosis'
nb_train_samples = 2000         #2000+rotation
nb_validation_samples = 150
#epochs  --  30/60/90/120
epochs = 50
#for epochs in [30,60,90,120]:
#    print("||||||----------|||||||")

#batch_size  --  8/12/16
batch_size = 8

#kernel_size -- 3/4/5
kernel_size=3
#filters -- 32/64

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

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('normalized_'+str(img_width)+'_'+str(epochs)+'_'+str(batch_size)+'_64_'+str(kernel_size)+'-arch3convpool3denseKERA2DROP.h5')



















'''
#original
model = Sequential()
model.add(Conv2D(32, kernel_size, input_shape=input_shape))
model.add(Activation('relu'))
#model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size))
model.add(Activation('relu'))
#model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size))
model.add(Activation('relu'))
#model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
#model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
'''


'''
#
model = Sequential()
model.add(Conv2D(64, kernel_size, input_shape=input_shape))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.1))

#
model.add(Conv2D(64, kernel_size, input_shape=input_shape))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.1))

#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.1))

#
model.add(Conv2D(64, kernel_size))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.1))

#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.1))

#
model.add(Dense(64))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(64))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(64))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.1))

model.add(Dropout(0.5))

#
model.add(Dense(64))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(64))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(64))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(64))
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.1))


model.add(Dense(1))
model.add(Activation('sigmoid'))
'''



