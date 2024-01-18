from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K, Input, Model
import scipy

img_width, img_height = 224, 224

train_data_dir = 'v_data/OCT2017/train'
validation_data_dir = 'v_data/OCT2017/test'
nb_train_samples = 5000
nb_validation_samples =90
epochs = 22
batch_size = 15

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Input layer
input_layer = Input(shape=input_shape)

# Layer 1
x = Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu')(input_layer)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

# Layer 2
x = Conv2D(256, kernel_size=(5, 5), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

# Layer 3
x = Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu')(x)

# Layer 4
x = Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu')(x)

# Layer 5
x = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

# Flatten and fully connected layers
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)

# Output layer
output_layer = Dense(4, activation='softmax')(x)

# Create and return the model
model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['CNV','DME', 'DRUSEN', 'NORMAL'])

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save('model_saved.h5')

