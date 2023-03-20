import hsv_color_picker
import utilis
import cv2



import numpy as np




 
cap = cv2.VideoCapture('vid.mp4')
# cap = cv2.VideoCapture(0)
cap.set(3, 256)
cap.set(4, 256)
frameCounter = 0


debug_mode = True

if debug_mode:
        utilis.hsv_window()
        utilis.initializeTrackbars([0,0,0,256])

while True:
    frameCounter +=1
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) ==frameCounter:
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        frameCounter=0
 
    _, img = cap.read()
    img = cv2.resize(img, (256,256))
    
    
    if debug_mode:
        masked_img = utilis.hsv_trackbar(img)
        points = utilis.valTrackbars()
        h, w, c = img.shape
        drawed_image = utilis.drawPoints(img, points)

    
        img_warp = utilis.warpImage(masked_img, points,w, h)
        
        hstack = np.hstack([img, masked_img,drawed_image, img_warp])
        print(img_warp)
        cv2.imshow('Wrap', hstack)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()