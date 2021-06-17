import cv2

car_classifier='cars.xml'
pedestrian_classifier='pedestrian1.xml'
# create an openCV image and import in gray
img = cv2.imread('car,ped.png')

# create trackers using classifiers using OpenCV
car_tracker = cv2.CascadeClassifier(car_classifier)
pedestrian_tracker=cv2.CascadeClassifier(pedestrian_classifier)
# detect cars
cars = car_tracker.detectMultiScale(img)
pedestrian= pedestrian_tracker.detectMultiScale(img)
# display the coordinates of different cars - multi dimensional array
print(cars)
print(pedestrian)

# draw rectangle around the cars
for (x,y,w,h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(img, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# draw rectangle around the pedestrian
for (x,y,w,h) in pedestrian:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
    cv2.putText(img, 'Human', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Finally display the image with the markings
cv2.imshow('my detection',img)

# wait for the keystroke to exit
cv2.waitKey(0)