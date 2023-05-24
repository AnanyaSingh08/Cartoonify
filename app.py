# coding=utf-8
import sys
import os, shutil
import glob
import re
import numpy as np
from collections import defaultdict
from scipy import stats
import cv2
from matplotlib import pyplot as plt

# Flask utils
from flask import Flask,flash, request, render_template,send_from_directory
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__, static_url_path='')
app.secret_key = os.urandom(24)

app.config['CARTOON_FOLDER'] = 'cartoon_img'
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/uploads/<filename>')
def upload_img(filename):
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/cartoon_images/<filename>')
def cartoon_img(filename):
    
    return send_from_directory(app.config['CARTOON_FOLDER'], filename)


def cartoonize(image):
    
    #convert image into cartoon-like image
    #image: input PIL image

    output = np.array(image)
    x, y, c = output.shape
    for i in range(c):
        output[:, :, i] = cv2.bilateralFilter(output[:, :, i], 5, 50, 50)
    
    edge = cv2.Canny(output, 100, 200)
    
   
    output = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)

    hists = []
    #H
    hist, _ = np.histogram(output[:, :, 0], bins=np.arange(180+1))
    hists.append(hist)
    #S
    hist, _ = np.histogram(output[:, :, 1], bins=np.arange(256+1))
    hists.append(hist)
    #V
    hist, _ = np.histogram(output[:, :, 2], bins=np.arange(256+1))
    hists.append(hist)

    C = []
    for h in hists:
        C.append(k_histogram(h))
    

    output = output.reshape((-1, c))
    for i in range(c):
        channel = output[:, i]
        index = np.argmin(np.abs(channel[:, np.newaxis] - C[i]), axis=1)
        output[:, i] = C[i][index]
    output = output.reshape((x, y, c))
    output = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)
    contours, _ = cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
    cv2.drawContours(output, contours, -1, 0, thickness=1)
    return output

def update_C(C, hist):
    
    
    
    while True:
        groups = defaultdict(list)

        #assign pixel values
        for i in range(len(hist)):
            if hist[i] == 0:
                continue
            d = np.abs(C-i)
            index = np.argmin(d)
            groups[index].append(i)

        new_C = np.array(C)
        for i, indice in groups.items():
            if np.sum(hist[indice]) == 0:
                continue
            new_C[i] = int(np.sum(indice*hist[indice])/np.sum(hist[indice]))
        if np.sum(new_C-C) == 0:
            break
        C = new_C
    return C, groups


def k_histogram(hist):
    
    
    
    alpha = 0.001              # p-value threshold for normaltest
    N = 80                      # minimun group size for normaltest
    C = np.array([128])

    while True:
        C, groups = update_C(C, hist)

        #start increase K if possible
        new_C = set()     
        for i, indice in groups.items():
            #if there are not enough values in the group, do not seperate
            if len(indice) < N:
                new_C.add(C[i])
                continue

            # judge whether we should seperate the centroid
            # by testing if the values of the group is under a
            # normal distribution
            z, pval = stats.normaltest(hist[indice])
            if pval < alpha:
                #not a normal dist, seperate
                left = 0 if i == 0 else C[i-1]
                right = len(hist)-1 if i == len(C)-1 else C[i+1]
                delta = right-left
                if delta >= 3:
                    c1 = (C[i]+left)/2
                    c2 = (C[i]+right)/2
                    new_C.add(c1)
                    new_C.add(c2)
                else:
                    # though it is not a normal dist, we have no
                    # extra space to seperate
                    new_C.add(C[i])
            else:
                # normal dist, no need to seperate
                new_C.add(C[i])
        if len(new_C) == len(C):
            break
        else:
            C = np.array(sorted(new_C))
    return C



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        
        f = request.files['file']
        style = request.form.get('style')
        print(style)
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        
        f.save(file_path)
        file_name=os.path.basename(file_path)
        
        # reading the uploaded image
        
        img = cv2.imread(file_path)
        cart_fname =file_name
        cartoonized = cartoonize(img)
        cartoon_path = os.path.join(
                basepath, 'cartoon_img', secure_filename(cart_fname))
        fname=os.path.basename(cartoon_path)
        print(fname)
        cv2.imwrite(cartoon_path,cartoonized)
        return render_template('predict.html',file_name=file_name, cartoon_file=fname)
           
    
  
if __name__ == '__main__':
        app.run(debug=True, host="localhost", port=8080)