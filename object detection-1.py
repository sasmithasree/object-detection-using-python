import cv2
img = cv2.imread("living hall.jpg")
img = cv2.resize(img,(600,600))
configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightpath='frozen_inference_graph.pb'
classNames=[]
classfile= 'coco.names'
with open(classfile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
net=cv2.dnn_DetectionModel(weightpath,configpath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)
classid,configs,bbox= net.detect(img,confThreshold=0.5)
print(classid,bbox)
sum=0
for classes ,con,box in zip(classid.flatten(),configs.flatten(),bbox):
    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
    cv2.putText(img,classNames[classes-1],(box[0]+10,box[1]+30),
    cv2.FONT_ITALIC,1,(0,0,255),2)
    print(classNames[classes-1])
    sum= sum+1
print("total objects detected= ",sum)
cv2.imshow("omnamasivaya",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


