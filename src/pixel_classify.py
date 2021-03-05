"""
File: pixel_classify.py
Author : Shaurya Chandhoke
Description: Helper file which contains the functions used for pixel classification
"""
import cv2
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
from tqdm import trange

from src.image_processing_helper import filePartitioner
from src.image_processing_helper import logger


def generateMask(image):
    """
    Helper function that separates sky pixels from non sky pixels

    :param image: The flattened training image
    :return: The indices for both sky and non sky pixels
    """
    skyIndices = np.where(np.all(image == 255, axis=1))[0]
    nonSkyIndices = np.where(~np.all(image == 255, axis=1))[0]
    return skyIndices, nonSkyIndices


def generateClusters(image, indices):
    """
    Helper function that utilizes the in-built K Means library to fit a model to the image

    :param image: The flattened training image
    :param indices: A numpy array representing key points of interest in the image
    :return: The fitted model
    """
    pixels = image[indices]
    model = KMeans(n_clusters=10)
    model.fit(pixels)
    return model


def paintImage(image, skyModel, nonSkyModel, desc):
    """
    Final part of the program where it determines whether or not to paint the specific pixel the unique color or not

    :param image: The flattened test image
    :param skyModel: K Means model fitted to determine sky pixels
    :param nonSkyModel: K Means model fitted to determine non sky pixels
    :param desc: The filename
    :return: The painted image
    """
    skyCentroids = skyModel.cluster_centers_
    nonskyCentroid = nonSkyModel.cluster_centers_
    iterator = image.shape[0]
    for i in trange(iterator, ncols=120, desc=desc, leave=False):
        pixel = image[i]
        skyIndex = distance.cdist(np.array([pixel]), skyCentroids).argmin()
        nonSkyIndex = distance.cdist(np.array([pixel]), nonskyCentroid).argmin()

        skyDistance = np.linalg.norm(skyCentroids[skyIndex] - pixel)
        nonSkyDistance = np.linalg.norm(nonskyCentroid[nonSkyIndex] - pixel)

        if skyDistance < nonSkyDistance:
            image[i, :] = [255, 0, 255]

    return image


def pixel_classifier(imageDir):
    """
    Entry point into the pixel classifier.

    The program will partition the image files into training and testing sets and then collect the differences
    between the masked image and original image to determine what is considered a sky.

    After the determination, the program will utilize an in-built K-Means algorithm to classify sky and non-sky pixels.

    Using the two unsupervised models, the program will iterate through each pixel in the test set to determine whether
    or not it is classified as a sky pixel or not. If so, it will be painted with the following RGB values:
         RGB: 255, 0, 255   (purple)

    Using the in-built K-Means algorithm allows for faster speed as it performs a series of approximations and
    gradient descent implementations to ensure a faster convergence.

    :param imageDir: List of files used for this problem
    :return: The painted test images as well as their file names
    """
    logger('2', 0, 'Performing Pixel Classification')

    trainingSet, testingSet = filePartitioner(imageDir)

    if '_mask' not in trainingSet[0]:
        temp = trainingSet[0]
        trainingSet[0] = trainingSet[1]
        trainingSet[1] = temp

    imageMask = cv2.imread(trainingSet[0], cv2.COLOR_BGR2RGB)
    trainingImage = cv2.imread(trainingSet[1], cv2.COLOR_BGR2RGB)

    logger('2a', 0, 'Aggregating sky and non-sky pixels', True)

    imageMask = imageMask.reshape(imageMask.shape[0] * imageMask.shape[1], 3)
    trainingImage = trainingImage.reshape(trainingImage.shape[0] * trainingImage.shape[1], 3)

    skyIndices, nonSkyIndices = generateMask(imageMask)

    logger('2a', 1, 'Aggregating sky and non-sky pixels', True, 'Num Sky Pixels: {}'.format(skyIndices.shape[0]),
           'Num Non-Sky Pixels: {}'.format(nonSkyIndices.shape[0]))
    logger('2b', 0, '[TRAINING] Generating clusters for sky and non-sky pixels', True)

    skyModel = generateClusters(trainingImage.copy(), skyIndices)
    nonSkyModel = generateClusters(trainingImage.copy(), nonSkyIndices)

    logger('2b', 1, '[TRAINING] Generating clusters for sky and non-sky pixels', True,
           'Sky Model Iterations: {}'.format(skyModel.n_iter_),
           'Non-Sky Model Iterations: {}'.format(nonSkyModel.n_iter_))

    logger('2c', 0, '[TESTING] Drawing sky pixels on test images', True)

    newImageList = []
    fileList = []
    for file in testingSet:
        image = cv2.imread(file, cv2.COLOR_BGR2RGB)
        originalShape = image.shape
        image = image.reshape(image.shape[0] * image.shape[1], 3)

        desc = file.split('/')[-1]
        fileList.append(desc)

        drawnImage = paintImage(image, skyModel, nonSkyModel, desc)
        drawnImage = drawnImage.reshape(originalShape)
        newImageList.append(drawnImage)

    logger('2c', 1, '[TESTING] Drawing sky pixels on test images', True, 'Images painted: {}'.format(len(newImageList)))
    logger('2', 1, 'Performing Pixel Classification')

    return newImageList, fileList
