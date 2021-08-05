import cv2
print(cv2.__version__)
img_path = "D:\OpenCV using python\humans.jpg"
img = cv2.imread(img_path)
# print(f"image shape is : {img.shape}")
# print(img)
cv2.imshow("In_RGB_Format", img)
cv2.waitKey()
cv2.destroyAllWindows()

im_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow("In_BGR_Format", im_bgr)
cv2.waitKey()
cv2.destroyAllWindows()
