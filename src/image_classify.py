"""
File: image_classify.py
Author : Shaurya Chandhoke
Description: Helper file which contains the functions used for image classification
"""
import cv2
import numpy as np
from prettytable import PrettyTable as Table
from tqdm import trange

from src.image_processing_helper import filePartitioner
from src.image_processing_helper import logger

# These variables are intentionally kept global for reusability and ease of use
binSpace = np.arange(0, 0, dtype=np.int32)
binCounter = np.zeros(shape=binSpace.shape, dtype=np.int32)
bins = 0


def binIndex(pixel):
    """
    A vectorized function used to categorize which is the closest bin the pixel in the image should belong to

    :param pixel: A single RGB pixel from an image
    :return: The closest bin index the pixel belongs to
    """
    for histBin, index in zip(binSpace, range(binSpace.shape[0])):
        binRange = (histBin, histBin + bins)
        if binRange[0] <= pixel < binRange[1]:
            return index


def histogramGenerator(image, file):
    """
    Helper function that transformed the image into its 1 x b histogram feature space

    :param image: The flattened image
    :param file: The image file
    :return: The transformed image
    """
    global binSpace, binCounter

    file = file.split('/')[-1]

    rgbChannels = [image[:, :, 2], image[:, :, 1], image[:, :, 0]]
    imageHistogram = []

    binSpace = np.arange(0, image.shape[0], bins, dtype=np.int32)
    binCounter = np.zeros(shape=binSpace.shape, dtype=np.int32)
    binIndexer = np.vectorize(binIndex)

    for index in trange(3, ncols=120, desc=file, leave=False):
        rgbChannel = rgbChannels[index]

        binIndexes = binIndexer(rgbChannel.ravel())
        np.add.at(binCounter, binIndexes, 1)

        imageHistogram.append(binCounter)
        binCounter = np.zeros(shape=binSpace.shape, dtype=np.int32)

    imageHistogram = np.asarray(imageHistogram)
    return imageHistogram


def classifyHistogram(histogram, histogramMatrix, labels):
    """
    Core classification function that will perform a 1-nearest neighbor algorithm to determine the predicted label

    :param histogram: The 1 x b image as a histogram space
    :param histogramMatrix: The training histogram matrix
    :param labels: The labels associated with each index of the histogram matrix
    :return: The predicted label
    """
    distances = []
    for feature in histogramMatrix:
        l2_norm = np.linalg.norm(histogram - feature)
        distances.append(l2_norm)

    distances = np.asarray(distances)
    closestIndex = np.argmin(distances)

    return labels[closestIndex]


def prettyPrint(trueLabels, predictedLabels, files, noshow):
    """
    Finalized step in the program where the results are tabulated into a pretty format and returned as a string

    :param trueLabels: The true labels of the image
    :param predictedLabels: The predicted labels of the image
    :param files: All the files tested
    :param noshow: Boolean representing whether or not to output the aggregated results to stdout
    :return: The tabularized result as a string
    """
    correctCount = 0
    table = Table(['File', 'True Class', 'Predicted Class'])

    for index in range(len(trueLabels)):
        file = files[index]
        trueLabel = trueLabels[index]
        predictedLabel = predictedLabels[index]
        table.add_row([file, trueLabel, predictedLabel])

        if trueLabel == predictedLabel:
            correctCount += 1

    if noshow is False:
        print(table)
        score = (correctCount / len(trueLabels)) * 100
        print('Accuracy: {:06.5f}%'.format(score))
        print("\r\n")

    return table.get_string()


def image_classifier(imageDir, numBins, verify, noshow):
    """
    Entry point into the image classifier.

    The program will partition the image files into training and testing sets and then perform RGB histogram
    aggregation for each image. Each RGB histogram will be flattened and appended to convert the image into a
    histogram space.

    Aggregating this over each training image will result in a:
        n x b
    histogram matrix, where n is the number of training images and b is the number of bins multiplied by 3 (for RGB)

    The test set will be transformed into a similar 1 x b dimension and the 1-nearest neighbor classifier will
    determine which feature it is closest to in relation to the histogram matrix. The closest distance will determine
    its predicted label.

    :param imageDir: List of files used for this problem
    :param numBins: Value representing a modified bin value to use for the histogram
    :param verify: Boolean representing whether or not to output an extra histogram verification step for each image
    :param noshow: Boolean representing whether or not to output the aggregated results to stdout
    :return: The aggregated results in a tabularized string format
    """
    global bins

    logger('1', 0, 'Performing Image Classification')
    logger('1a', 0, 'Partitioning image set into training and testing images', True)
    trainingSet, testingSet = filePartitioner(imageDir)
    logger('1a', 1, 'Partitioning image set into training and testing images', True,
           'Train_Size: {}'.format(len(trainingSet)), 'Test_Size: {}'.format(len(testingSet)))

    logger('1b', 0, '[TRAINING] Generating histograms for each image rgb channel', True)

    if verify:
        logger('1-Verification', 0, 'Verification Step', True)

    histogramMatrix = []
    labelContainer_training = []
    for index in range(len(trainingSet)):
        file = trainingSet[index]

        trueLabel_training = file.split('/')[-1].split('_')[0]
        labelContainer_training.append(trueLabel_training)

        image = cv2.imread(file, cv2.COLOR_BGR2RGB)
        bins = image.shape[0] // numBins

        rgbHistograms = histogramGenerator(image, file)
        if verify:
            print('\t\tHistogram matrix should have 3 rows: {} -> {}'.format(rgbHistograms.shape,
                                                                             rgbHistograms.shape[0] == 3))
        rgbHistograms = rgbHistograms.flatten()
        histogramMatrix.append(rgbHistograms)

    histogramMatrix = np.asarray(histogramMatrix)

    if verify:
        logger('1-Verification', 1, 'Verification Step', True)

    logger('1b', 1, '[TRAINING] Generating histograms for each image rgb channel', True, 'Bins: {}'.format(numBins))
    logger('1c', 0, '[TESTING] Evaluating test images against histogram collection', True)

    labelContainer_testing = []
    predictedContainer = []
    fileContainer = []
    for index in range(len(testingSet)):
        file = testingSet[index]
        image = cv2.imread(file, cv2.COLOR_BGR2RGB)

        rgbHistograms = histogramGenerator(image, file).flatten()

        file = file.split('/')[-1]
        trueLabel_testing = file.split('_')[0]
        predictedLabel = classifyHistogram(rgbHistograms, histogramMatrix, labelContainer_training)

        labelContainer_testing.append(trueLabel_testing)
        predictedContainer.append(predictedLabel)
        fileContainer.append(file)

    output = prettyPrint(labelContainer_testing, predictedContainer, fileContainer, noshow)

    logger('1c', 1, '[TESTING] Evaluating test images against histogram matrix', True)
    logger('1', 1, 'Performing Image Classification')

    return output
