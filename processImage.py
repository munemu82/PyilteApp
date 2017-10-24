import cv2
"""
SIFT Parameters
-------------------------
nfeatures	
The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast)

nOctaveLayers	
The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.

contrastThreshold	
The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.

edgeThreshold	
The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).

sigma	
The sigma of the Gaussian applied to the input image at the octave #0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.

SURF Parameters
------------------------
hessianThreshold	
Threshold for hessian keypoint detector used in SURF.

nOctaves	
Number of pyramid octaves the keypoint detector will use.

nOctaveLayers	
Number of octave layers within each octave.

extended	
Extended descriptor flag (true - use extended 128-element descriptors; false - use 64-element descriptors).
upright	
Up-right or rotated features flag (true - do not compute orientation of features; false - compute orientation).


"""
def processImage(imgPath, processType):
    if processType =='grayscale':
        img = cv2.imread(imgPath, 1)
        finalImage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif processType =='grayscaledHistEq':
        img = cv2.imread(imgPath, 1)
        finalImage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #perform histogram equalization
        finalImage = cv2.equalizeHist(finalImage)
    elif processType =='colorHistEq':
        img = cv2.imread(imgPath)
        colorImg = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        colorImg[:, :, 0] = cv2.equalizeHist(colorImg[:, :, 0])
        finalImage = cv2.cvtColor(colorImg, cv2.COLOR_YUV2BGR)
    elif processType=='siftFeatures':
        img = cv2.imread(imgPath, 1)
        img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        (kps, descs) = sift.detectAndCompute(img2, None)
        #print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
        finalImage = cv2.drawKeypoints(img2, kps, img.copy())
        #print(type(finalImage))
    elif processType == 'surfFeatures':
        img = cv2.imread(imgPath, 1)
        img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        surf = cv2.xfeatures2d.SURF_create(400)
        surf.setHessianThreshold(100)
        (kps, descs) = surf.detectAndCompute(img2, None)
        # print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
        finalImage = cv2.drawKeypoints(img2, kps, img.copy())
    elif processType == 'cannyDetector':
        img = cv2.imread(imgPath, 1)
        finalImage = cv2.Canny(img, 100, 200)
    else:
        finalImage = cv2.imread(imgPath, 1)
        #print('Path or image does not exist')
    return finalImage

def resizeImage(imgPath, cols, rows):
    img = cv2.imread(imgPath)
    newImage = cv2.resize(img,(cols,rows))
    return newImage