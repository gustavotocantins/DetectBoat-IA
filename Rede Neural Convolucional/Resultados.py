import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import re
import PIL
from PIL import Image
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten
import random

def jpeg_to_8_bit_greyscale(path, maxsize):
        img = Image.open(path).convert('L')   
        WIDTH, HEIGHT = img.size
        if WIDTH != HEIGHT:
                m_min_d = min(WIDTH, HEIGHT)
                img = img.crop((0, 0, m_min_d, m_min_d))
        img.thumbnail(maxsize, PIL.Image.ANTIALIAS)
        return np.asarray(img)

def load_image_dataset(path_dir, maxsize):
        images = []
        labels = []
        os.chdir(path_dir)
        for file in glob.glob("*.jpg"):
                img = jpeg_to_8_bit_greyscale(file, maxsize)

                if re.match('balsa.*', file):
                        images.append(img)
                        labels.append(0)

                elif re.match('canoa.*', file):
                        images.append(img)
                        labels.append(1)

                elif re.match('catraia.*', file):
                        images.append(img)
                        labels.append(2)

                elif re.match('ferry boat.*', file):
                        images.append(img)
                        labels.append(3)   

                elif re.match('iate.*', file):
                        images.append(img)
                        labels.append(4)                    

                elif re.match('navio.*', file):
                        images.append(img)
                        labels.append(5)

                elif re.match('popopo.*', file):
                        images.append(img)
                        labels.append(6)

                elif re.match('rabeta.*', file):
                        images.append(img)
                        labels.append(7)

                elif re.match('veleiro.*', file):
                        images.append(img)
                        labels.append(8)

                elif re.match('voadeira.*', file):
                        images.append(img)
                        labels.append(9)
                        
        return (np.asarray(images), np.asarray(labels))

maxsize = 100, 100
def display_images(images, labels):
        plt.figure(figsize=(80, 80))
        grid_size = min(55, len(images))
        for i in range(grid_size):
                plt.subplot(11, 5, i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(images[i], cmap=plt.cm.binary)
                plt.xlabel(class_names[labels[i]])

plt.show()
epocas = [1,5,10,30,60,100,240,500]

for n_ep in epocas:
        (train_images, train_labels) = load_image_dataset(r'C:\Users\gusta\Desktop\Laboratorio\Redes Neurais\Barcos\treino', maxsize)
        (test_images, test_labels) = load_image_dataset(r'C:\Users\gusta\Desktop\Laboratorio\Redes Neurais\Barcos\teste', maxsize)

        #EMBARALHAR AS IMAGENS

        # Crie uma lista de índices na mesma ordem da lista original
        # Crie uma lista de índices na mesma ordem da lista original
        indices = list(range(len(train_labels)))

        # Embaralhe os índices
        np.random.shuffle(indices)

        # Use os índices embaralhados para embaralhar as duas listas
        train_images = train_images[indices]
        train_labels = train_labels[indices]

        class_names = [ 'balsa', 'canoa', 'catraia','ferry boat','iate','navio','popopo','rabeta','veleiro','voadeira']
        display_images(test_images, test_labels)
        train_images = train_images / 255
        test_images = test_images / 255


#CRIAR LOOP
        model = Sequential() # 28x28
        model.add(Conv2D(64, kernel_size=(3,3), activation='relu',
        input_shape=(100, 100, 1) )) #20x24x24
        model.add(MaxPooling2D(pool_size=(2,2))) #20x12x12
        model.add(Conv2D(128, kernel_size=(3,3), activation='relu')) #40x8x8
        model.add(MaxPooling2D(pool_size=(2,2))) #40x4x4
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        model.compile(optimizer=keras.optimizers.Adam(0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

        print(train_images.shape)
        model.fit(train_images, train_labels, epochs=n_ep,verbose=False)
        test_loss, test_acc = model.evaluate(test_images, test_labels)

        acerto = str(test_acc*100)
        print(f'Épocas: {n_ep} | accuracy: {acerto[:4]}%')