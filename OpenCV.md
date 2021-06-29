import cv2

face_cascade = cv2.CascadeClassifier( f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml" )

img =cv2.imread("photo.jpg")

gray_img = cv2.cvtColor( img , cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img , scaleFactor=1.05, minNeighbors=5 )

for x,y,w,h in faces:
    img=cv2.rectangle( img , (x,y) , (x+h , y+h) ,(0,255,0) ,3 )

cv2.imshow("Gray" , img)

cv2.waitKey(0)

cv2.destroyAllWindows()


### Vedio Capture 

 import cv2 ,time 

vedio = cv2.VedioCapture(0)

check , frame = vedio.read()

time.sleep(3)

cv2.imshow("Capturing" , frame )

cv2.waitKey(0)

vedio.release()

cv2.destroyAllWindows()


import cv2 , time 

vedio = cv2.VedioCapture(0)

a=1 

while true :
    
    a = a+1
    
    check,frame = vedio.read()
    
    print(frame)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Capturing",gray)
    
    key=cv2.waitkey(1)
    
    if key == ord('q'):
        
         break
print(a)

vedio.release()

cv2.destroyAllWindows
    
    
## Multiple Frame detection 

 import cv2 , time 
 vedio = cv2.VedioCapture(0)
 a=1 
 while true :
     a = a+1
     check,frame = vedio.read()
     print(frame)
     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     cv2.imshow("Capturing",gray)
     key=cv2.waitkey(1)
     if key == ord('q'):
           break
print(a)

vedio.release()

cv2.destroyAllWindows

### Motion Detector 

import cv2 , time 

first_frame = None 

while true:
    check , frame = vedio.read(0)
    
    gray=cv2.cvtCOLOR(frame , cv2.COLOR_BGR2GRAY)
    
    gray=cv2.GuassianBlur(gray,(21,21),0)
    
    if first_frame is None :
        first_frame = gray
        continue
    delta_frame = absdiff(first_frame , gray)
    
    thresh_delta = cv2.threshold(delta_frame , 30 , 255 , cv2.THRESH_BINARY[1] )
    
    thresh_delta = cv2.dilate(thresh_delta , None , iteration = 0 )
    
    (_,cnts,_) = cv2.findContours( thresh_delta.copy() , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE) 
    
    for contours in cnts :
        if cv2.contourArea(contour) < 1000:
            continue
        (x , y , w , h) = cv2.boundingRect(contour)
        
        cv2.rectangle( frame , (x,y) , (x+w , y+h) ,(0,255,0) ,3 )
    
    cv2.imshow('frame' , frame)
    
    cv2.imshow('Capturing' , gray)
    
    cv2.imshow('delta' delta_frame )
    
    cv2.imshow('Thresh' thresh_delta)
    if Key == ord('q'):
        break
        
vedio.release()

cv2.destroyAllWindows
    
    #emotion detector

import os  
import cv2  
import numpy as np  
from keras.models import model_from_json  
from keras.preprocessing import image  
  
#load model  
model = model_from_json(open("model.json", "r").read())  
#load weights  
model.load_weights('model.h5')  
  
  
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  
  
t=[] 
cap=cv2.VideoCapture(0)  
  
while (len(t)<2000):  
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image  
    if not ret:  
        continue  
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  
  
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)  
  

    for (x,y,w,h) in faces_detected:  
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)  
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
        roi_gray=cv2.resize(roi_gray,(48,48))  
        img_pixels = image.img_to_array(roi_gray)  
        img_pixels = np.expand_dims(img_pixels, axis = 0)  
        img_pixels /= 255  
  
        predictions = model.predict(img_pixels)  
  
        #find max indexed array  
        max_index = np.argmax(predictions[0])  
  
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')  
        predicted_emotion = emotions[max_index]  
        t.append(0.3*predictions[0][0]+0.05*predictions[0][1]+0.09*predictions[0][2]+0.005*predictions[0][3]+0.4*predictions[0][4]+0.003*predictions[0][5]+0.152*predictions[0][6])
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  
  
    resized_img = cv2.resize(test_img, (1000, 700))  
    cv2.imshow('Facial emotion analysis ',resized_img)  
  
  
  
    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed  
        break  
  
cap.release()  
cv2.destroyAllWindows  

from scipy.signal import filtfilt
from scipy.signal import butter,lfilter
from scipy.signal import freqs,freqz
import numpy as np
from scipy.fft import fft,ifft,fftfreq
import pandas as pd
import matplotlib.pyplot as plt 

from scipy import signal
%matplotlib inline
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import ipywidgets as widgets
from obspy.signal.trigger import recursive_sta_lta,carl_sta_trig,plot_trigger#,trigger_onset
from obspy.core.trace import Trace
from collections import deque 

def trigger_onset(charfct, thres1, thres2, max_len=9e99, max_len_delete=False):    
    ind1 = np.where(charfct > thres1)[0]
    if len(ind1) == 0:
        return []
    ind2 = np.where(charfct > thres2)[0]
    #
    on = deque([ind1[0]])
    of = deque([-1])
    # determine the indices where charfct falls below off-threshold
    ind2_ = np.empty_like(ind2, dtype=bool)
    ind2_[:-1] = np.diff(ind2) > 10
    # last occurence is missed by the diff, add it manually
    ind2_[-1] = True
    of.extend(ind2[ind2_].tolist())
    on.extend(ind1[np.where(np.diff(ind1) > 1)[0] + 1].tolist())
    # include last pick if trigger is on or drop it
    if max_len_delete:
        # drop it
        of.extend([1e99])
        on.extend([on[-1]])
    else:
        # include it
        of.extend([ind2[-1]])
    #
    pick = []
    while on[-1] > of[0]:
        while on[0] <= of[0]:
             on.popleft()
        while of[0] < on[0]:
             of.popleft()
        if of[0] - on[0] > max_len:
            if max_len_delete:
                on.popleft()
                continue
            of.appendleft(on[0] + max_len)
        pick.append([on[0], of[0]])
    return np.array(pick, dtype=np.int64)
    
    @widgets.interact(sta_time=(0.01,10,0.01),lta_time=(5,120,1),thr_on=(1,20,0.1),thr_off=(0.1,8,0.1))
def alpha_calc(sta_time,lta_time,thr_on,thr_off):
    alpha=np.zeros(len(t))
    sta=0.
    csta=1./sta_time
    clta=1./lta_time
    icsta=1-csta
    iclta=1-clta
    lta=1e-99
    for i in range(1,len(t)):
      a=(t[i]**2)
      sta=a*csta+icsta*sta
      lta=a*clta+iclta*lta
      if(i<lta_time):
            alpha[i]=0
      else:
             alpha[i]=sta/lta
    plt.figure(figsize=(10,10))
    plt.plot(alpha)
    cft=alpha
    trace=Trace(np.array(t))
    df = trace.stats.sampling_rate
    npts = trace.stats.npts
    k = np.arange(npts, dtype=np.float32) / df
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(211)
    ax1.plot(k*0.02,t , 'k')
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(k*0.02, cft, 'k')
    cft1=np.array(cft)
    on_off = np.array(trigger_onset(cft1, thr_on, thr_off))
    i, j = ax1.get_ylim()
    try:
        ax1.vlines(on_off[:, 0]*0.02 / df, i, j, color='r', lw=2,
                   label="Trigger On")
        ax1.vlines(on_off[:, 1]*0.02 / df, i, j, color='b', lw=2,
                   label="Trigger Off")
        ax1.legend()
    except IndexError:
        pass
    ax2.axhline(thr_on, color='red', lw=1, ls='--')
    ax2.axhline(thr_off, color='blue', lw=1, ls='--')
    ax2.set_xlabel("Time after %s [s]" % trace.stats.starttime.isoformat())
    fig.suptitle(trace.id)
    fig.canvas.draw()
    plt.show()
    
    @widgets.interact(sta_time=(0.01,10,0.01),lta_time=(5,120,1),thr_on=(1,20,0.1),thr_off=(0.1,8,0.1))
def alpha_calc(sta_time,lta_time,thr_on,thr_off):
    alpha=np.zeros(len(t))
    sta=0.
    csta=1./sta_time
    clta=1./lta_time
    icsta=1-csta
    iclta=1-clta
    lta=1e-99
    for i in range(1,len(X)):
      a=(t[i]**2)
      sta=a*csta+icsta*sta
      lta=a*clta+iclta*lta
      if(i<lta_time):
            alpha[i]=0
      else:
             alpha[i]=sta/lta
    plt.figure(figsize=(10,10))
    plt.plot(alpha)
    cft=alpha
    trace=Trace(np.array(y))
    df = trace.stats.sampling_rate
    npts = trace.stats.npt
    k = np.arange(npts, dtype=np.float32) / df
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(211)
    ax1.plot(k*0.02,t , 'k')
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(t*0.02, cft, 'k')
    cft1=np.array(cft)
    on_off = np.array(trigger_onset(cft1, thr_on, thr_off))
    i, j = ax1.get_ylim()
    try:
        ax1.vlines(on_off[:, 0]*0.02 / df, i, j, color='r', lw=2,
                   label="Trigger On")
        ax1.vlines(on_off[:, 1]*0.02 / df, i, j, color='b', lw=2,
                   label="Trigger Off")
        ax1.legend()
    except IndexError:
        pass
    ax2.axhline(thr_on, color='red', lw=1, ls='--')
    ax2.axhline(thr_off, color='blue', lw=1, ls='--')
    ax2.set_xlabel("Time after %s [s]" % trace.stats.starttime.isoformat())
    fig.suptitle(trace.id)
    fig.canvas.draw()
    plt.show()
    
    
    






