import cv2
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# img = cv2.imread('D:/OpenCV using python/human pic.webp')
# cv2.imshow('test', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# cv2.imshow('Grey_image', gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# faces = face_cascade.detectMultiScale(gray)
# print(faces)

# for(x,y,w,h) in faces:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
# cv2.imshow('face',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def fun_to_detect(img_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
    eyes = eye_cascade.detectMultiScale(gray)
    for(x,y,w,h) in eyes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,127,0), 2)
    cv2.imshow('face',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

fun_to_detect("D:/OpenCV using python/humans.jpg")
