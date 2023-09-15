import numpy as np
import matplotlib.pyplot as plt
import glob, os
import re
import PIL
from PIL import Image
import pickle 

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
                plt
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(images[i], cmap=plt.cm.binary)
                plt.xlabel(class_names[labels[i]])

plt.show()
epocas = [1,5,10,30,60,100,240,500]
(train_images, train_labels) = load_image_dataset(r'C:\Users\gusta\OneDrive\Documentos\GitHub\DetectBoat-IA\treino', maxsize)
(test_images, test_labels) = load_image_dataset(r'C:\Users\gusta\OneDrive\Documentos\GitHub\DetectBoat-IA\teste', maxsize)
train_images = train_images / 255
test_images = test_images / 255
class_names = [ 'balsa', 'canoa', 'catraia','ferry boat','iate','navio','popopo','rabeta','veleiro','voadeira']

filename = r'C:\Users\gusta\OneDrive\Documentos\GitHub\DetectBoat-IA\Rede Neural Multicamada\modelorede.pkl'
with open(filename, 'rb') as file:
    model = pickle.load(file)

predictions = model.predict(test_images)
classe = np.argmax(predictions, axis = 1)
print(class_names[classe])