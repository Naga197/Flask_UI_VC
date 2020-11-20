from flask import Flask,render_template,url_for,request,Response,redirect
import flask_sqlalchemy
from flask_sqlalchemy import SQLAlchemy
import cv2
import time
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import time
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
from elapsedtimer import ElapsedTimer
import pathlib
from pathlib import Path
import videocaptioning_predict_for_a_single_video as vp
import videocaptioning_predict_sv_vgg16 as vvg



app=Flask(__name__)

BASE_DIR = 'static/Videos' 
path_prj= 'd:/Users/Naga/video_captions/Resnet_files'
feat_dir='Resnet_feat_dir'
caption_file='test2.csv'
pred_vfile='0bSz70pYAP0_5_15'
print("current_dir:",path_prj)
path_vgg='d:/Users/Naga/video_captions/vgg16_files'
vgg_feat_dir='vgg16_feat'
cap_file='test.csv'



 
def gen(req_path):
    abs_path = os.path.join(BASE_DIR, req_path) 
    cap = cv2.VideoCapture(abs_path)
    
    # Re  ad until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, (0,0), fx=0.75, fy=0.75) 
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            #time.sleep(0.1)
        else: 
            break 
    cap.release()
    cv2.destroyAllWindows()   

@app.route("/")
def index():
    
    return render_template('index.html')
 

@app.route("/video_feed",methods=["GET","POST"])
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    if request.method=="GET":
        filename=request.args.get('vfile')
        return Response(gen(filename),
            mimetype='multipart/x-mixed-replace; boundary=frame')
       
           
     
@app.route("/predict",methods=["GET","POST"])
def predict():
    """Video streaming route. Put this in the src attribute of an img tag."""
    if request.method=="GET":
        filename=request.args.get('vfile')
        pred_vfile=filename.split('.')[0]
        mytest=vp.VideoCaptioning(path_prj,caption_file,feat_dir)
        predict=mytest.inference(pred_vfile)
        mytest2=vvg.VideoCaptioning(path_vgg,cap_file,vgg_feat_dir)
        predict_vgg=mytest2.inference(pred_vfile)
        return render_template('predict.html',value1=predict,value2=predict_vgg)

        
                

  

if __name__=="__main__":
    app.run(debug=True)    