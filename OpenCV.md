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
    
    






