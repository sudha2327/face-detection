import cv2
import os

dataset="dataset"

name="sugan"

path=os.path.join(dataset,name)

if not os.path.isdir(path):
    os.mkdir(path)

(width,height)=(130,100)

count=1

algo="haarcascade_frontalface_default.xml"

harcas=cv2.CascadeClassifier(algo)

cam=cv2.VideoCapture(0)

while count<31:
    print(count)
    _,img=cam.read()

    grayimage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=harcas.detectMultiScale(grayimage,1.3,4)

    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        faceOnly=grayimage[y:y+h,x:x+h]
        resize=cv2.resize(faceOnly,(width,height))
        cv2.imwrite("%s/%s.jpg"%(path,count),faceOnly)
        count+=1
       
    cv2.imshow("facedetection",img)
    key=cv2.waitKey(10)

    if key==27:
        break;
cam.release()
cv2.destroyAllWindows()
