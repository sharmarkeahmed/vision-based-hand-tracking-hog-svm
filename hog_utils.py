import cv2
hog = cv2.HOGDescriptor()
im = cv2.imread(sample)
h = hog.compute(im)