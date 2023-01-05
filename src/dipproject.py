# -*- coding: utf-8 -*-
"""DIPProject.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cDF_3eXmr8pzBaWXLYnNWystuifUYMXD

# Import Libraries
"""

from google.colab import drive
drive.mount("/content/drive")

IMAGES_FOLDER_PATH = "/content/drive/My Drive/DIP_Images/imagesProject/"



"""# Read Image"""

img = cv.imread(os.path.join(IMAGES_FOLDER_PATH,"chess2.png"))
imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)

# Plotting Original
fig1 = plt.imshow(imgRGB)
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Original Image")
plt.show()

img1 = imgRGB.copy()
imgRGBN = 255-img1
# Plotting Negative
fig1 = plt.imshow(imgRGBN)
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Negative Image")
plt.show()

"""# RGB Separation"""

r,g,b = imgRGB[:,:,0],imgRGB[:,:,1],imgRGB[:,:,2]

imRed,imGreen,imBlue = imgRGB.copy(),imgRGB.copy(),imgRGB.copy()
imRed[:,:,0],imRed[:,:,1],imRed[:,:,2] = r,r,r
imGreen[:,:,0],imGreen[:,:,1],imGreen[:,:,2] = g,g,g
imBlue[:,:,0],imBlue[:,:,1],imBlue[:,:,2] = b,b,b

# Plotting Red
fig1 = plt.imshow(imRed)
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Red Channel Image")
plt.show()
# Plotting Green
fig1 = plt.imshow(imGreen)
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Green Channel Image")
plt.show()
# Plotting Blue
fig1 = plt.imshow(imBlue)
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Blue Channel Image")
plt.show()

"""# Noise Removal - Median Blur"""

imRedMed = cv.medianBlur(imRed,11)
imGreenMed = cv.medianBlur(imGreen,11)

# Plotting Median Blurred
fig1 = plt.imshow(imRedMed)
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Median Blurred Red Image")
plt.show()

# Plotting Median Blurred
fig1 = plt.imshow(imGreenMed)
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Median Blurred Green Image")
plt.show()

"""# Edge Detection before thresholding"""

edgesRed = cv.Canny(imRed,100,200);
edgesGreen = cv.Canny(imGreen,100,200);

# Plotting Edges
fig1 = plt.imshow(edgesRed, cmap='gray')
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Edges in Red Image")
plt.show()

# Plotting Edges
fig1 = plt.imshow(edgesGreen, cmap='gray')
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Edges in Green Image")
plt.show()

edgesT = edgesRed.copy()
edgesT = np.maximum(edgesRed,edgesGreen)

# Plotting Edges
fig1 = plt.imshow(edgesT, cmap='gray')
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Edges in Red-Green Image")
plt.show()

"""# Thresholding"""

_,imRedMedT = cv.threshold(imRedMed[:,:,0],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
_,imGreenMedT = cv.threshold(imGreenMed[:,:,0],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# Plotting Thresholded
fig1 = plt.imshow(imRedMedT, cmap='gray')
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Thresholded Red Image")
plt.show()

# Plotting Thresholded
fig1 = plt.imshow(imGreenMedT, cmap='gray')
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Thresholded Green Image")
plt.show()

"""# Edge Detection after Thresholding"""

edgesRedT = cv.Canny(imRedMedT,100,200);
edgesGreenT = cv.Canny(imGreenMedT,100,200);

# Plotting Edges
fig1 = plt.imshow(edgesRedT, cmap='gray')
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Edges in Red Image")
plt.show()

# Plotting Edges
fig1 = plt.imshow(edgesGreenT, cmap='gray')
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Edges in Green Image")
plt.show()

edgesT = edgesRedT.copy()
edgesT = np.maximum(edgesRedT,edgesGreenT)

# Plotting Edges
fig1 = plt.imshow(edgesT, cmap='gray')
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Edges in Red-Green Image")
plt.show()

"""# Hough Transform"""

lines = cv.HoughLines(edgesT, 1, np.pi/180, 200, None, 0, 0)
edgesPT = cv.cvtColor(edgesT, cv.COLOR_GRAY2BGR)

if lines is not None:
  for i in range(0, len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
    pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
    cv.line(edgesPT, pt1, pt2, (255,0,0), 3, cv.LINE_AA)

# Plotting Lines
fig1 = plt.imshow(edgesPT)
plt.title("Lines in Red-Green Image")
plt.show()

pts_src = [[420,980],[420,1070],[500,1000],[500,1090]]
pts_dst = [[420,1000],[420,1090],[500,1000],[500,1090]]
pts_src = np.array(pts_src)
pts_dst = np.array(pts_dst)
h,status = cv.findHomography(pts_src, pts_dst);

im_dst = cv.warpPerspective(edgesPT, h, [edgesPT.shape[0],edgesPT.shape[1]])
# Plotting Homography
fig1 = plt.imshow(im_dst[:1100,:,:])
plt.title("Homography")
plt.show()

def getCorners(self, image):
        """Find subpixel chessboard corners in image."""
        temp = image
        ret, corners = cv.findChessboardCorners(temp, (7, 7),flags=cv.CALIB_CB_ADAPTIVE_THRESH +cv.CALIB_CB_FAST_CHECK +cv.CALIB_CB_NORMALIZE_IMAGE)
        return corners

out = getCorners(edgesT,edgesT);
print(out)

"""# White Piece Color Isolation"""

# Plotting Red
fig1 = plt.imshow(imRed)
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Red Channel Image")
plt.show()
# Plotting Green
fig1 = plt.imshow(imGreen)
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Green Channel Image")
plt.show()
# Plotting Blue
fig1 = plt.imshow(imBlue)
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Blue Channel Image")
plt.show()

"""## Blurring"""

imBlueMed = cv.medianBlur(imBlue,11)

# Plotting Median Blurred
fig1 = plt.imshow(imBlueMed)
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Median Blurred Blue Image")
plt.show()

"""## Dilation"""

kernel = np.ones((5,5),np.uint8)
imBlueDil = cv.dilate(imBlueMed,kernel)
imBlueDil = cv.dilate(imBlueDil,kernel)


# Plotting Dilated
fig1 = plt.imshow(imBlueDil)
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Dilated Blue Image")
plt.show()

"""## Thresholding"""

_,whitePieces = cv.threshold(imBlueDil[:,:,0],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

threshold = 50
"""
for i in range(0,imBlueDil.shape[0]):
  for j in range(0,imBlueDil.shape[1]):
    for k in range(0,imBlueDil.shape[2]):
      if(imBlueDil[i,j,k]>threshold):
        whitePieces[i,j,k] = 255
      else:
        whitePieces[i,j,k] = 0
"""
# Plotting Thresholded
fig1 = plt.imshow(whitePieces,cmap='gray')
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("White Pieces")
plt.show()

"""# Black Piece Color Isolation"""

im1 = np.maximum(imRedMedT,imGreenMedT)

# Plotting combined RG
fig1 = plt.imshow(im1, cmap='gray')
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Combined Red Green Thresholded Image")
plt.show()

# Plotting Blue
fig1 = plt.imshow(imBlue)
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Blue Channel Image")
plt.show()

"""## Blurring"""

imBlueMed = cv.medianBlur(imBlue,11)

# Plotting Median Blurred
fig1 = plt.imshow(imBlueMed)
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Median Blurred Blue Image")
plt.show()

"""## Thresholding for ideal range"""

blackPieces1 = imBlueMed.copy()

threshold = 80

for i in range(0,imBlueDil.shape[0]):
  for j in range(0,imBlueDil.shape[1]):
    for k in range(0,imBlueDil.shape[2]):
      if(imBlueDil[i,j,k]>threshold):
        blackPieces1[i,j,k] = 0
      elif(imBlueDil[i,j,k]>10):
        blackPieces1[i,j,k] = 255

# Plotting Thresholded
fig1 = plt.imshow(blackPieces1)
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Thresholded Black Pieces")
plt.show()

"""## Dilation"""

kernel = np.ones((5,5),np.uint8)
blackPieces = cv.dilate(blackPieces1,kernel)
blackPieces = cv.dilate(blackPieces,kernel)


# Plotting Dilated
fig1 = plt.imshow(blackPieces)
fig1.axes.get_xaxis().set_visible(False)
fig1.axes.get_yaxis().set_visible(False)
plt.title("Black Pieces")
plt.show()

"""## Sending th inputs to CNN"""



"""## CNN"""

import os

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras_preprocessing.image import ImageDataGenerator
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np

classes = ['bishop', 'pawn', 'knight', 'rook']
num_classes = len(classes)


class Model:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='valid', input_shape=(300, 300, 3), activation='relu'))
        self.model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.1))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(num_classes, activation='softmax'))

        self.model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])
        plot_model(self.model, to_file='model.png', show_shapes=True)
        self.model.summary()

    def train(self, batch_size=64, epochs=720):
        """
        Trains the model. Keras' ImageDataGenerators are used to flow data from directories
        data/train and data/validation where the subdirectory names are the data labels.
        """
        train_datagen = ImageDataGenerator(
            shear_range=0.2,
            rotation_range=180,
            height_shift_range=0.1,
            width_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False)

        test_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow_from_directory(
            'data\\train',
            target_size=(300, 300),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True)

        validation_generator = test_datagen.flow_from_directory(
            'data\\validation',
            target_size=(300, 300),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True)

        fit_generator = self.model.fit_generator(
            train_generator,
            steps_per_epoch=1543 / batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=314 / batch_size)

        self.visualize(epochs, fit_generator)
        self.__save_weights('%s_epochs_model_weights.h5' % epochs)

    @staticmethod
    def visualize(epochs, fit_generator):
        """
        Plots all the applicable data from the training history.
        """
        for key in fit_generator.history.keys():
            plt.style.use('ggplot')
            plt.figure()
            plt.plot(np.arange(0, epochs), fit_generator.history[key], label=key)
            plt.title(key)
            plt.xlabel('Epoch #')
            plt.ylabel(key)
            plt.legend(loc='lower left')
            plt.savefig(os.path.join('plots', '{}_model_{}_plot.png'.format(epochs, key)))

    def predict(self, weights_file, img):
        """
        Evaluates the given image with the given weights.
        """
        if self.model.weights is None:
            self.__load_weights(weights_file)
        score = self.model.evaluate()
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def __save_weights(self, weights_file):
        """
        Saves the model weights to the provided file.
        """
        self.model.save_weights(os.path.join('weights', weights_file))

    def __load_weights(self, weights_file):
        """
        Loads the weights file provided.
        """
        self.model.load_weights(os.path.join('weights', weights_file))

def mpredict():
  return '-'

n = 24
m = Model()
m.train()

"""## Final test and assemble"""

def get__piece(Pieces, start_row, end_row, row_step, start_col, end_col, col_step):
    y = []
    for i in range(start_row, end_row, row_step):
        for j in range(start_col, end_col, col_step):
            square = Pieces[i : i+row_step, j : j+col_step]
            if(np.all(square > 0)):
                y.append(mpredict())
    
    return y

start_row = 130
end_row = 870
start_col = 300
end_col = 1500
row_step = 100
col_step = 100

# Get White pieces
y = get__piece(whitePieces, start_row, end_row, row_step, start_col, end_col, col_step)
y = np.asarray(y)
file = open("file1.txt", "w+")

# Get Black pieces
y = get__piece(blackPieces, start_row, end_row, row_step, start_col, end_col, col_step)
y = np.asarray(y)
file = open("file2.txt", "w+")

with open('file1.txt') as fp:
    data = fp.read()

with open('file2.txt') as fp:
    data2 = fp.read()

data += "\n"
data += data2
  
with open ('file3.txt', 'w') as fp:
    fp.write(data)