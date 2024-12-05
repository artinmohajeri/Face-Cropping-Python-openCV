import cv2

image = cv2.imread('img/Elon.jpg')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

if len(faces) == 0:
    raise ValueError("No faces found in the image.")
elif len(faces) > 1:
    raise ValueError("There are more than one faces in the image.")

x, y, w, h = faces[0]
cropped_face = image[y:y+h, x:x+w]
cv2.imwrite('cropped_face.jpg', cropped_face)
