import cv2
import numpy as np

class HOGFeaturesExtract:

    #define the initial constructor
    def __init__(self, winSize, blockSize, blockStride, cellSize, nbins, signedGradients):
        # store the number of points and radius
        self.winSize = winSize
        self.blockSize = blockSize
        self.blockStride = blockStride
        self.cellSize = cellSize
        self.nbins = nbins
        self.signedGradients = signedGradients

    def describe(self, image):
        #define default variables
        derivAperture = 1
        winSigma = -1.
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64

        #setup HOG Descriptor
        # hog = cv2.HOGDescriptor(self.winSize, self.blockSize, self.blockStride, self.cellSize, self.nbins,
        #                         derivAperture, winSigma, histogramNormType, L2HysThreshold,
        #                         gammaCorrection, nlevels, self.unsignedGradients)
        hog = cv2.HOGDescriptor(self.winSize, self.blockSize, self.blockStride, self.cellSize, self.nbins, derivAperture, winSigma,
                          histogramNormType, L2HysThreshold, gammaCorrection, nlevels, self.signedGradients)
        #Compute HOG features and descriptors on image
        #hogDescVector = hog.compute(image)
        winStride = (8, 8)
        padding = (8, 8)
        locations = ((10, 20),)
        hogDescVector = hog.compute(image, winStride, padding, locations)
        hogDescVector = hogDescVector.ravel()

        return hogDescVector