from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import os
from matplotlib import pyplot as plt
from PIL import Image

#problems: overfitting
#if we change, be sure to also change which model we are loading and saving into
#keras2 file - targsize = 80
#keras3 file - targsize = 120

batch_size= 24
num_classes = 928
targ_size = 80

def cnn_model(model, training):
    if training:
        data_generator = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.2,
        )
        train_generator = data_generator.flow_from_directory(
            'complete-pokemon-image-dataset',
            target_size=(targ_size, targ_size),
            batch_size=batch_size,
            color_mode='rgb',
            subset="training"
        )
        val_generator = data_generator.flow_from_directory(
            'complete-pokemon-image-dataset',
            target_size=(targ_size, targ_size),
            batch_size=batch_size,
            color_mode='rgb',
            subset="validation"
        )
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=12, cooldown=6, rate=0.6, min_lr=1e-18, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=24, verbose=1)

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        hist = model.fit_generator(train_generator, validation_data=val_generator, validation_steps=300, epochs=5,
                            steps_per_epoch=500, shuffle=True, callbacks=[reduce_lr,early_stop])

        plt.plot(hist.history['acc'])
        plt.plot(hist.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = 'keras_poke5_trained_model.h5'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)


def predict_this(model, this_img):
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    im = this_img.resize((targ_size,targ_size)) # size expected by network
    img_array = np.array(im)

    img_array = img_array/255. # rescale pixel intensity as expected by network
    img_array = np.expand_dims(img_array, axis=0) # reshape from (160,160,3) to (1,160,160,3)
    pred = model.predict(img_array)
    return np.argmax(pred, axis=1).tolist()[0]

def driver():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(targ_size, targ_size, 3,), activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=4))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), padding='same',  activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation="softmax"))
    model.load_weights('saved_models/keras_poke5_trained_model.h5')

    cnn_model(model, True)
    classes = [_class for _class in os.listdir('complete-pokemon-image-dataset/')]
    classes.sort() #
    classes = classes[1:]
    print(classes[predict_this(model, Image.open('output.png'))])

driver()


