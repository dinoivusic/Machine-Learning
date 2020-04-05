#!/usr/bin/env python
# coding: utf-8

# In[41]:


#Simple program for detecting faces and eyes

#import OpenCV 
import cv2 

# The Haar Classifiers
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

img = cv2.imread('./HTS.jpg')
faces = face_cascade.detectMultiScale(img) # you can add 2 more optional params

#print number of faces we found
print('Faces found:', len(faces))
print('The image height, width and channel: ', img.shape)
print('The coordinates of each face detected: ', faces)


# In[42]:


#iterate over coordinates and mark them with rectangle
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    roi_face = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_face)
    for (eye_x, eye_y, eye_w, eye_h) in eyes:
        cv2.rectangle(roi_face, (eye_x, eye_y), (eye_x+eye_w, eye_y + eye_h), (255,0,0), 2)

font = cv2.FONT_HERSHEY_SIMPLEX
text = cv2.putText(img, 'Face Detected',(0, img.shape[0]), font, 2, (255, 255, 255), 2)


# In[43]:


#show image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

