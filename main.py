from numpy.lib.type_check import imag
import tensorflow as tf
from keras.preprocessing import image #<-- Load and turn image in array
from keras.preprocessing.image import ImageDataGenerator 
import cv2
import numpy as np
import os


print('### Preparando dados ###')
FormatoDados = ImageDataGenerator(
    rescale=1./255,
)
    
path = os.path.abspath('')
pathTraining = f"{path}\\training_set\\training_set"
pathTest = f"{path}\\test_set\\test_set"


dadosDeTreino = FormatoDados.flow_from_directory(
    pathTraining,
    target_size = (64,64),
    batch_size = 32,
    class_mode = 'binary'
)

DadosDeTeste = FormatoDados.flow_from_directory(
    pathTest,
    target_size=(64,64),
    batch_size = 32,
    class_mode = 'binary'
)





print('### Criando modelo ###')
modelo = tf.keras.models.Sequential()

modelo.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu', input_shape=[64,64,3]))
modelo.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

modelo.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
modelo.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

modelo.add(tf.keras.layers.Flatten())

modelo.add(tf.keras.layers.Dense(units=128,activation='relu'))
modelo.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

print('Compilando Modelo')
modelo.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

modelo.fit(x=dadosDeTreino, validation_data= DadosDeTeste, epochs=3)


while True:

    ipt = int(input('> '))
    coisa = cv2.imread(f'{ipt}.jpg')

    test_image = image.load_img(f'{ipt}.jpg',target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)

    result = modelo.predict(test_image)

    if result[0][0] == 1:
        print(' > Dog')
    if result[0][0] == 0:
        print(' > Cat')


    cv2.imshow('Foto', coisa)
    cv2.waitKey(0)




























