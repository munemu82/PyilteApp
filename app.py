from flask import Flask, request, redirect, url_for, render_template, send_file
from processImage import processImage, resizeImage           #import our custom image processing function
from LocalBinaryPattern import LocalBinaryPatterns
from HOGFeaturesExtract import HOGFeaturesExtract
import traceback
import os
import json
import glob
from uuid import uuid4
import cv2
import numpy as np
import pandas as pd

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global processingType, errors, errors2
    global procImgFullPath, filename
    global featureVectors, featuresFilePath

    featureVectors = []  #define feature vectors list and read the source image

    """Handle the upload of a file."""
    form = request.form
    
    # Create a unique "session ID" for this particular batch of uploads.
    upload_key = str(uuid4())

    # Is the upload using Ajax, or a direct POST by the form?
    is_ajax = False
    if form.get("__ajax", None) == "true":
        is_ajax = True

    # Target folder for these uploads. 
    target = "static/uploads/{}".format(upload_key)
    target_processed = target+"/processed/"
    try:
        os.mkdir(target)
        os.mkdir(target_processed)
    except:
        if is_ajax:
            return ajax_response(False, "Couldn't create upload directory: {}".format(target))
        else:
            return "Couldn't create upload directory: {}".format(target)

    print("=== Form Data ===")
    processingType = request.form['processingType']

    for key, value in list(form.items()):
        print(key, "=>", value)
    for upload in request.files.getlist("file"):
        filename = upload.filename.rsplit("/")[0]
        destination = "/".join([target, filename])
        print("Accept incoming file:", filename)
        print("Save it to:", destination)
        upload.save(destination)
        procImgFullPath = target_processed + "processed_" + filename;
        #perform image processing
        if processingType == "imageresize":
            imgNewWidth = request.form['resWidth']
            imgNewHeight = request.form['resHeight']
            cv2.imwrite(procImgFullPath, resizeImage(destination, int(float(imgNewWidth)),int(float(imgNewHeight))))
        elif processingType == "LBP":
            numOfPoints = int(float(request.form['numPoints']))
            radiusVal = int(float(request.form['radius']))
            #create LBP descriptor object
            desc = LocalBinaryPatterns(numOfPoints, radiusVal)
            image = cv2.imread(destination)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hist = desc.describe(gray)
            #add feature vectors to the list
            featureVectors.append(hist)
        elif processingType == "HOG":
            winSize = (int(float(request.form['winSize1'])), int(float(request.form['winSize2'])))
            cellSize = (int(float(request.form['cellSize1'])), int(float(request.form['cellSize2'])))
            blockSize = (int(float(request.form['blockSize1'])), int(float(request.form['blockSize2'])))
            blockStride = (int(float(request.form['blockStride1'])), int(float(request.form['blockStride2'])))
            numOfBins = int(float(request.form['numOfBins']))
            signedGradients = bool(request.form['numOfBins'])
            #Create HOG Descriptor object
            hogDesc =  HOGFeaturesExtract(winSize, blockSize, blockStride, cellSize, numOfBins, signedGradients)
            # winSize = (64, 64)
            # blockSize = (16, 16)
            # blockStride = (8, 8)
            # cellSize = (8, 8)
            # nbins = 9
            derivAperture = 1
            winSigma = 4.
            histogramNormType = 0
            L2HysThreshold = 2.0000000000000001e-01
            gammaCorrection = 0
            nlevels = 64
            # hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, numOfBins, derivAperture, winSigma,
            #                         histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)

            image = cv2.imread(destination,0)
            # winStride = (8, 8)
            # padding = (8, 8)
            # locations = ((10, 20),)
            # finalHogVec = hog.compute(image, winStride, padding, locations)
            # finalHogVec = finalHogVec.ravel()
            try:
                finalHogVec = hogDesc.describe(image)
            except:
                errors = traceback.format_exc()
            featureVectors.append(finalHogVec)
        else:
            cv2.imwrite(procImgFullPath, processImage(destination, processingType))

    #Common code for all feature extractions
    if featureVectors:      #Check if list of feature vectors is not empty
        #convert numpy array to pandas dataframe
        numpyFeatVectors = np.asarray(featureVectors)  # convert python list of features to numpy
        print(numpyFeatVectors.shape)
        try:
            df = pd.DataFrame(numpyFeatVectors)
            featuresFilePath = target_processed+"/"+processingType+".csv"
            df.to_csv(featuresFilePath)
        except:
            errors2 = traceback.format_exc()

    if is_ajax:
        return ajax_response(True, upload_key)
    else:
        return redirect(url_for("upload_complete", uuid=upload_key))

@app.route("/files/<uuid>")
def upload_complete(uuid):
    """The location we send them to at the end of the upload."""
    global file
    global btn
    btn = "None"
    # Get their files.
    root = "static/uploads/{}".format(uuid)
    rootProcessed = root+"/processed"
    print("The processed image folder: "+rootProcessed)

    if not os.path.isdir(root):
        return "Error: UUID not found!"

    files = []
    processedFiles = []
    processeduuid = uuid+"/processed"

    #add uploaded images to the list
    for file in glob.glob("{}/*.*".format(root)):
        fname = file.split(os.sep)[-1]
        files.append(fname)
        procFname = "processed_"+fname
        processedFiles.append(procFname)
    if (processingType =="LBP") or (processingType =="HOG"):
        btn = "download.html"
    #pass the list of files (uploaded and processed) to the html template (files.html)
    return render_template("files.html",
        uuid=uuid,
        files=files,
        processedFiles=processedFiles,
        processeduuid=processeduuid,
        processingType=processingType,
        btn=btn,
    )


def ajax_response(status, msg):
    status_code = "ok" if status else "error"
    return json.dumps(dict(
        status=status_code,
        msg=msg,
    ))

@app.route('/about/')
def about():
    return render_template("about.html")

@app.route('/imageview/')
def imageview():
    return render_template("imageview.html")

@app.route('/documentation/')
def documentation():
    return render_template("documentation.html")
@app.route("/download")
def download():
    #extract file name from featuresFilePath
    #featuresFileName = featuresFilePath.rsplit("/")[0]
    return send_file(featuresFilePath, attachment_filename=processingType+".csv", as_attachment=True)

if __name__ == '__main__':
    app.debug=True
    app.run()