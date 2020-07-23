#importing libraries
import cv2
import numpy as np

#initializing haarcascade fullbody classifier
classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

#initializing video file
video = cv2.VideoCapture('video.mp4')

#looping while video is open
while video.isOpened():
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fullbody = classifier.detectMultiScale(gray, 1.2, 3)
    
    #inner loop for making boxes around people
    for (x,y,w,h) in fullbody:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 3)
        cv2.imshow('Haar Cascade Full Body Classifier', frame)

    #break condition for terminating the video
    if cv2.waitKey(1) == 13: 
        break

video.release()
cv2.destroyAllWindows()
