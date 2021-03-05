"""
File: visual_recognizer.py
Author : Shaurya Chandhoke
Description: Command line script that takes a directory path as input and processes the images within them as output
"""
import argparse
import collections
import os
import time

import cv2

from src.image_classify import image_classifier
from src.pixel_classify import pixel_classifier


def output_processing(tableData, paintedImages, fileNames, nosave, noshow, timeElapsed):
    """
    The final stages of the program. This function will display the images and/or write them to files as well as
    provide an execution time.

    :param tableData: The tabularized results of problem 1
    :param paintedImages: The painted images of problem 2
    :param fileNames: The file names of problem 2
    :param nosave: Boolean indicating whether or not to save the results to the ./out directory
    :param noshow: Boolean indicating whether or not to show the results
    :param timeElapsed: Total run time of the program
    """
    if (noshow is True) and (nosave is True):
        print("(BOTH FLAGS ON) Recommend disabling either --nosave or --quiet to capture processed images")
        return 0

    print("=" * 40)
    print("Rendering Images...")

    if noshow is False:
        print("(DISPLAY ON) The ESC key will close all pop ups")
        for i in range(len(fileNames)):
            file = fileNames[i]
            image = paintedImages[i]
            cv2.imshow(file, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if nosave is False:
        print("(IMAGE SAVE ON) Images and report are being written to the ./out/ folder")
        for i in range(len(fileNames)):
            file = fileNames[i]
            image = paintedImages[i]
            cv2.imwrite('./out/painted_{}'.format(file), image)

        with open('./out/problem1_report.txt', 'w+') as outputStream:
            title = '{:^55}'.format('Problem 1 Report')
            border = '=' * 55
            message = title + '\n' + border + '\n'

            outputStream.write(message)
            outputStream.write(tableData)

    message = "You may want to rerun the program with the --verify flag or the --help flag for more options to fine " \
              "tune the program."

    print("(DONE): " + message)
    print("=" * 40)
    print("Time to Process Image: {} seconds.".format(timeElapsed))


def start(problem1_Imgset, problem2_Imgset, bins, verify, noshow):
    """
    Starter function responsible for beginning the process for obtaining edges

    :param problem1_Imgset: List containing all files used for problem 1
    :param problem2_Imgset: List containing all files used for problem 2
    :param bins: Value representing modified bin value to use for problem 1
    :param verify: Boolean indicating whether or not to output extra verification step
    :param noshow: Boolean indicating whether or not to print output of problem 1 to stdout
    :return: The tabularized results of problem 1, the painted images and file names for problem 2
    """
    print("Please wait, processing images and returning output...\n")

    table = image_classifier(problem1_Imgset, bins, verify, noshow)
    paintedImages, fileNames = pixel_classifier(problem2_Imgset)

    return table, paintedImages, fileNames


def main():
    """
    Beginning entry point into the visual recognition program.
    It will first perform prerequisite steps prior to starting the intended program.
    Upon parsing the command line arguments, it will trigger the start function
    """
    '''
    As per the homework assignment, we are to process images found only in the following directories:
        ImClass
        sky

    By saving these names into a variable, these two directories can be located in a folder with other items.
    This program will be able to extract them from the directory if they exist regardless.    
    '''
    dirNames = ['ImClass', 'sky']

    # Reusable message variables
    ADVICE = "rerun with the (-h, --help) for more information."

    # Start cli argparser
    temp_msg = "Given the path to a directory containing the folders: \'ImClass\' and \'sky\', this program will " \
               "perform the recognition processes."
    parser = argparse.ArgumentParser(prog="visual_recognizer.py", description=temp_msg,
                                     usage="%(prog)s [imgpath] [flags]")

    temp_msg = "The directory path containing the two required directories. The path can be relative or absolute."
    parser.add_argument("dirpath", help=temp_msg, type=str)

    temp_msg = "The number of histogram bins. Default is 8"
    parser.add_argument("-b", "--bins", help=temp_msg, type=int, default=8)

    temp_msg = "If passed, prints an extra verification step to ensure each pixel is counted exactly 3 times during " \
               "image classification. By default, verification occurs silently"
    parser.add_argument("-v", "--verify", help=temp_msg, default=False, action="store_true")

    temp_msg = "If passed, the images will not be written to a file. By default, images are written."
    parser.add_argument("-n", "--nosave", help=temp_msg, action="store_true")

    temp_msg = "If passed, the images will not be displayed. By default, the images will be displayed."
    parser.add_argument("-q", "--quiet", help=temp_msg, action="store_true")

    # Obtain primary CLI arguments
    args = parser.parse_args()
    dirpath = args.dirpath
    bins = args.bins
    verify = args.verify
    nosave = args.nosave
    noshow = args.quiet

    # Begin error checking params and checking the validity of the directory
    if os.path.exists(dirpath):
        cleanedPath = os.path.abspath(dirpath)
        dirNames = [os.path.join(cleanedPath, name) for name in dirNames]

        items = [os.path.join(cleanedPath, item) for item in os.listdir(dirpath)]
        dirs = list(filter(lambda item: os.path.isdir(item), items))

        if collections.Counter(dirs) != collections.Counter(dirNames):
            print("Error: Cannot find required folders within this directory.\nPlease check if the path is correct "
                  "or " + ADVICE)
            return -1
    else:
        print("Error: Cannot find directory.\nPlease check if the path is correct or " + ADVICE)
        return -1

    if dirs[0] not in dirNames[0]:
        temp = dirs[0]
        dirs[0] = dirs[1]
        dirs[1] = temp

    problem1_Imgs = [os.path.join(dirs[0], file) for file in os.listdir(dirs[0])]
    problem2_Imgs = [os.path.join(dirs[1], file) for file in os.listdir(dirs[1])]

    START_TIME = time.time()

    table, paintedImages, fileNames = start(problem1_Imgs, problem2_Imgs, bins, verify, noshow)

    END_TIME = time.time() - START_TIME

    output_processing(table, paintedImages, fileNames, nosave, noshow, END_TIME)


if __name__ == "__main__":
    main()
