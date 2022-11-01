import cv2
import pytesseract


img = cv2.imread("poe.png")

text = pytesseract.image_to_string(img,lang="eng")
print(text)



cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()



