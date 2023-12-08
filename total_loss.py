import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.ops.numpy_ops import np_config
import sys
import numpy as np
from keras.applications.vgg19 import VGG19
np_config.enable_numpy_behavior()
from pgt_block_48_v26 import pgt_block_48_v26


################ vgg #########################################################################
selected_layers = ['block1_conv1', 'block2_conv1',"block3_conv1" ,'block4_conv1','block5_conv4']
#selected_layer_weights = [1.0, 4.0 , 4.0 , 8.0 , 16.0]


vgg = VGG19(weights='imagenet', include_top=False, input_shape=(None,None,3))
vgg.trainable = False
outputs = [vgg.get_layer(l).output for l in selected_layers]
model = tf.keras.Model(vgg.input, outputs)



@tf.function
def perceptual_loss(input_image , reconstruct_image, features):
    features += 5
    selected_layer_weights = [0, 0 , 0 , 0 , features]
    #print(features)

    h1_list = model(input_image)
    h2_list = model(reconstruct_image)

    rc_loss = 0.0
    for h1, h2, weight in zip(h1_list, h2_list, selected_layer_weights):
        h1 = K.batch_flatten(h1)
        h2 = K.batch_flatten(h2)
        rc_loss = rc_loss + weight * K.sum(K.square(h1 - h2), axis=-1)
    return rc_loss
    




def custom_loss(y_true, y_pred):
    
    
    rgb_ypred = []
    rgb_ytrue = []
    vgg_loss = 0
    ori_loss = 0
    feature_loss = 0
    tmp_loss = 0
    

    ##batch size = 5
    for i in range(5):
      tmp = y_pred[i, :, :, 0].reshape(36,36,1)
      tmp = tf.image.grayscale_to_rgb(tmp)
      rgb_ypred = tf.expand_dims(tmp, 0)


      FeaturesTerminations, FeaturesBifurcations = extract_minutiae_features(y_true[i, :, :]*255, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False)
      features = len(FeaturesTerminations) + len(FeaturesBifurcations)

      tmp = y_true[i, :, :].reshape(36,36,1)
      tmp = tf.convert_to_tensor(tmp)
      tmp = tf.image.grayscale_to_rgb(tmp)
      rgb_ytrue = tf.expand_dims(tmp, 0)
      
      vgg_loss += perceptual_loss(rgb_ytrue, rgb_ypred, features)
      

    return vgg_loss

def custom_loss_wrapper(input_tensor):
  
    def custom_loss_1(y_true, y_pred):

      mse_channel_0 = keras.losses.MeanSquaredError()(y_true[:,:,:,0], y_pred[:,:,:,0])
      mse_channel_1 = keras.losses.MeanSquaredError()(y_true[:,:,:,1], y_pred[:,:,:,1])
      
    
      lapKernel = K.constant([[-1, -1],[-1, -1],[-1, -1],[-1, -1],[8, 8],[-1, -1],[-1, -1],[-1, -1],[-1, -1]], shape = [3, 3, 1, 2])
    
    
      trueLap = K.conv2d(y_true.astype('float32'), lapKernel)
      predLap = K.conv2d(y_pred.astype('float32'), lapKernel)

      lap = K.square(trueLap - predLap)

      ssim_channel_0 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,0], axis=-1), tf.expand_dims(y_pred[:,:,:,0], axis=-1), 1.0))
      ssim_channel_1 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,1], axis=-1), tf.expand_dims(y_pred[:,:,:,1], axis=-1), 1.0))
      
      loss_channel_0 = 0.1 * mse_channel_0 + 0.2 * lap[:,:,:,0] + 0.7 * ssim_channel_0
      loss_channel_1 = 0.1 * mse_channel_1 + 0.2 * lap[:,:,:,1] + 0.7 * ssim_channel_1

      

      ## discriminator loss
      customloss = custom_loss(input_tensor, y_pred)
      
       

      return 0.00000001*customloss + 0.7*loss_channel_0 + 0.3*loss_channel_1 

      
    return custom_loss_1


import cv2
import numpy as np
import skimage.morphology
from skimage.morphology import convex_hull_image, erosion
from skimage.morphology import square
import math
from google.colab.patches import cv2_imshow

class MinutiaeFeature(object):
    def __init__(self, locX, locY, Orientation, Type):
        self.locX = locX
        self.locY = locY
        self.Orientation = Orientation
        self.Type = Type

class FingerprintFeatureExtractor(object):
    def __init__(self):
        self._mask = []
        self._skel = []
        self.minutiaeTerm = []
        self.minutiaeBif = []
        self._spuriousMinutiaeThresh = 10

    def setSpuriousMinutiaeThresh(self, spuriousMinutiaeThresh):
        self._spuriousMinutiaeThresh = spuriousMinutiaeThresh

    def __skeletonize(self, img):
        img = np.uint8(img > 128)
        self._skel = skimage.morphology.skeletonize(img)
        self._skel = np.uint8(self._skel) * 255
        self._mask = img * 255

    def __computeAngle(self, block, minutiaeType):
        angle = []
        (blkRows, blkCols) = np.shape(block)
        CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
        if (minutiaeType.lower() == 'termination'):
            sumVal = 0
            for i in range(blkRows):
                for j in range(blkCols):
                    if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
                        angle.append(-math.degrees(math.atan2(i - CenterY, j - CenterX)))
                        sumVal += 1
                        if (sumVal > 1):
                            angle.append(float('nan'))
            return (angle)

        elif (minutiaeType.lower() == 'bifurcation'):
            (blkRows, blkCols) = np.shape(block)
            CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
            angle = []
            sumVal = 0
            for i in range(blkRows):
                for j in range(blkCols):
                    if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
                        angle.append(-math.degrees(math.atan2(i - CenterY, j - CenterX)))
                        sumVal += 1
            if (sumVal != 3):
                angle.append(float('nan'))
            return (angle)

    def __getTerminationBifurcation(self):
        self._skel = self._skel == 255
        (rows, cols) = self._skel.shape
        self.minutiaeTerm = np.zeros(self._skel.shape)
        self.minutiaeBif = np.zeros(self._skel.shape)

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if (self._skel[i][j] == 1):
                    block = self._skel[i - 1:i + 2, j - 1:j + 2]
                    block_val = np.sum(block)
                    if (block_val == 2):
                        self.minutiaeTerm[i, j] = 1
                    elif (block_val == 4):
                        self.minutiaeBif[i, j] = 1

        self._mask = convex_hull_image(self._mask > 0)
        self._mask = erosion(self._mask, square(5))  # Structuing element for mask erosion = square(5)
        self.minutiaeTerm = np.uint8(self._mask) * self.minutiaeTerm

    def __removeSpuriousMinutiae(self, minutiaeList, img):
        img = img * 0
        SpuriousMin = []
        numPoints = len(minutiaeList)
        D = np.zeros((numPoints, numPoints))
        for i in range(1,numPoints):
            for j in range(0, i):
                (X1,Y1) = minutiaeList[i]['centroid']
                (X2,Y2) = minutiaeList[j]['centroid']

                dist = np.sqrt((X2-X1)**2 + (Y2-Y1)**2)
                D[i][j] = dist
                if(dist < self._spuriousMinutiaeThresh):
                    SpuriousMin.append(i)
                    SpuriousMin.append(j)

        SpuriousMin = np.unique(SpuriousMin)
        for i in range(0,numPoints):
            if(not i in SpuriousMin):
                (X,Y) = np.int16(minutiaeList[i]['centroid'])
                img[X,Y] = 1

        img = np.uint8(img)
        return(img)

    def __cleanMinutiae(self, img):
        self.minutiaeTerm = skimage.measure.label(self.minutiaeTerm, connectivity=2)
        RP = skimage.measure.regionprops(self.minutiaeTerm)
        self.minutiaeTerm = self.__removeSpuriousMinutiae(RP, np.uint8(img))

    def __performFeatureExtraction(self):
        FeaturesTerm = []
        self.minutiaeTerm = skimage.measure.label(self.minutiaeTerm, connectivity=2)
        RP = skimage.measure.regionprops(np.uint8(self.minutiaeTerm))

        WindowSize = 2  # --> For Termination, the block size must can be 3x3, or 5x5. Hence the window selected is 1 or 2
        FeaturesTerm = []
        for num, i in enumerate(RP):
            (row, col) = np.int16(np.round(i['Centroid']))
            block = self._skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
            angle = self.__computeAngle(block, 'Termination')
            if(len(angle) == 1):
                FeaturesTerm.append(MinutiaeFeature(row, col, angle, 'Termination'))

        FeaturesBif = []
        self.minutiaeBif = skimage.measure.label(self.minutiaeBif, connectivity=2)
        RP = skimage.measure.regionprops(np.uint8(self.minutiaeBif))
        WindowSize = 1  # --> For Bifurcation, the block size must be 3x3. Hence the window selected is 1
        for i in RP:
            (row, col) = np.int16(np.round(i['Centroid']))
            block = self._skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
            angle = self.__computeAngle(block, 'Bifurcation')
            if(len(angle) == 3):
                FeaturesBif.append(MinutiaeFeature(row, col, angle, 'Bifurcation'))
        return (FeaturesTerm, FeaturesBif)

    def extractMinutiaeFeatures(self, img):
        self.__skeletonize(img)

        self.__getTerminationBifurcation()

        self.__cleanMinutiae(img)

        FeaturesTerm, FeaturesBif = self.__performFeatureExtraction()
        return(FeaturesTerm, FeaturesBif)

    def showResults(self, FeaturesTerm, FeaturesBif):
        
        (rows, cols) = self._skel.shape
        DispImg = np.zeros((rows, cols, 3), np.uint8)
        DispImg[:, :, 0] = 255*self._skel
        DispImg[:, :, 1] = 255*self._skel
        DispImg[:, :, 2] = 255*self._skel

        for idx, curr_minutiae in enumerate(FeaturesTerm):
            row, col = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
            skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))

        for idx, curr_minutiae in enumerate(FeaturesBif):
            row, col = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
            skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0))
        
        #cv2_imshow(DispImg)
        cv2.waitKey(0)

    def saveResult(self, FeaturesTerm, FeaturesBif):
        (rows, cols) = self._skel.shape
        DispImg = np.zeros((rows, cols, 3), np.uint8)
        DispImg[:, :, 0] = 255 * self._skel
        DispImg[:, :, 1] = 255 * self._skel
        DispImg[:, :, 2] = 255 * self._skel

        for idx, curr_minutiae in enumerate(FeaturesTerm):
            row, col = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
            skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))

        for idx, curr_minutiae in enumerate(FeaturesBif):
            row, col = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
            skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0))
        #cv2.imwrite('result.png', DispImg)

def extract_minutiae_features(img, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False):
    feature_extractor = FingerprintFeatureExtractor()
    feature_extractor.setSpuriousMinutiaeThresh(spuriousMinutiaeThresh)
    if (invertImage):
        img = 255 - img;

    FeaturesTerm, FeaturesBif = feature_extractor.extractMinutiaeFeatures(img)
    #print(len(FeaturesTerm))

    if (saveResult):
        feature_extractor.saveResult(FeaturesTerm, FeaturesBif)

    if(showResult):
        feature_extractor.showResults(FeaturesTerm, FeaturesBif)

    return(FeaturesTerm, FeaturesBif)