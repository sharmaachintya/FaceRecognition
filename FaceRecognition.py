import cv2
import numpy
import os
import urllib.request
import imutils

'''
url='http://192.168.29.218:8080/shot.jpg'  #For mobile cam
'''

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_file = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
face_cascade=cv2.CascadeClassifier(haar_file)
datasets='datasets'
print("Training...")
(images,labels,names,id)=([],[],{},0)

for (subdirs,dirs,files) in os.walk(datasets):
    for subdir in dirs:
        names[id]=subdir
        subjectpath=os.path.join(datasets,subdir)
        for filename in os.listdir(subjectpath):
            path=subjectpath+'/'+filename
            label=id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id=id+1

(images,labels)=[numpy.array(lis) for lis in [images,labels] ]
print(images,labels)
(width,height)=(130,100)
model=cv2.face.LBPHFaceRecognizer_create()
#model=cv2.face.FisherFaceRecognizer_create()

model.train(images,labels)

webcam=cv2.VideoCapture(0)
cnt=0

while True:
    (_,img)=webcam.read()
    '''
    imgPath=urllib.request.urlopen(url)                                     #For opening the url of IP cam
    imgNp=numpy.array(bytearray(imgPath.read()), dtype=numpy.uint8)         #For gettinf the data form IP cam
    img=cv2.imdecode(imgNp,-1)                                              #For converting that data into image
    img=imutils.resize(img,width=450)                                       #For resizing that image
    '''
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,x+h),(255,255,0),2)
        face=gray[y:y+h,x:x+w]
        face_resize=cv2.resize(face,(width,height))

        prediction=model.predict(face_resize)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        if prediction[1]<800:
            cv2.putText(img,'%s-%.0f'%(names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255))
            print(names[prediction[0]])
            cnt=0
        else:
            cnt=cnt+1
            cv2.putText(img,'Unknown',(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
            if(cnt>100):
                print("Unknown person")
                cv2.imwrite("Unkown.jpg",img)
                cnt=0
    '''           
    cv2.imshow("Camera Feed",img)  #For IP cam
    if ord('q')==cv2.waitKey(1):   #For IP cam
        6exit(0)                    #For IP cam
    '''
    cv2.imshow('FaceRecognition',img)
    key=cv2.waitKey(10)
    if key == 27:
        break
webcam.release()
cv2.destroyAllWindows()
