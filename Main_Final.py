import cv2
import numpy as np
import time
from matplotlib import pyplot as plt


cap=cv2.VideoCapture(0)

while True:
    cap.read=frame.rat()
    frame=cv2.resize(frame,None,fx=1.0,fy=1.0, interpolation=cv2.INTER_AREA)
cv2.imshow ('input',frame)
c=cv2.waitKey(1)
    

cap.release()
cap.destroyAllWindows()




    
vid=cv2.VideoCapture(0)
ret,frame= vid.read()
r = cv2.selectROI(frame) # select ROI by user  
imCrop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]# cropped image

track_window = (r[0],r[1],r[2],r[3])#initial location of hand
xp=r[0]
yp=r[1]
hsv_roi =  cv2.cvtColor(imCrop, cv2.COLOR_BGR2HSV)# convert ROI from RGB to HSV color space
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))# Th on cropped image to eliminate some pixels

roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])# calc histogram of HUE
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)# normalize histogram 


framep=frame

cv2.imshow("Image", imCrop)
cv2.waitKey(0)
vid=cv2.VideoCapture(0)

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )# criteria for stop 
cnt=0
kernel = np.ones((51,51),np.float)


while True:
    cnt+=1
    ret,frame= vid.read()

    # calculation of  background subtracted image  
    if cnt>=1:
        frame1=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame1=frame1.astype('float')
        
        framep=cv2.cvtColor(framep, cv2.COLOR_BGR2GRAY)
        framep=framep.astype('float')
        
        diff=abs(frame1-framep)
        diff[diff<=10]=0
        diff[diff>10]=1
        diff = cv2.dilate(diff,kernel,iterations = 1)
        
    framep=frame
 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    
    hsv[:,:,0]=np.multiply(hsv[:,:,0], diff)# multiplication of backgorund subtracted image in hue channel of frame
   
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)# computing a probablity of each pixel to represent hand area
 
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)# meanshift
    x,y,w,h = track_window

    if abs(x-xp)+abs(y-yp)>15:
        cv2.putText(frame,  " Hand is moving", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2) # put text

    cv2.rectangle(frame, (x,y), (x+w,y+h), 200,1)
    cv2.imshow('Frame',frame)
    
    xp=x
    yp=y
    if cv2.waitKey(1) & 0XFF== ord('q'):
        break

    
vid.release()

cv2.destroyAllWindows()
