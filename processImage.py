import cv2

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

        print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
        finalImage = cv2.drawKeypoints(img2, kps, img.copy())
        print(type(finalImage))
    else:
        print('Path or image does not exist')
    return finalImage