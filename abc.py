import cv2

# 1. Load Haarcascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Read the image
img = cv2.imread('sumit.jpg')  # kisi image ka path do
img = cv2.resize(img,(500,500))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale

# 3. Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# 4. Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 5. Show result
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
