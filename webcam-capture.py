import cv2 
from pathlib import Path

key = cv2. waitKey(1)
webcam = cv2.VideoCapture(1)
while True:
    try:
        check, frame = webcam.read()
        # print(check) #prints true as long as the webcam is running
        # print(frame) #prints matrix values of each framecd 
        cv2.circle(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)), 5, (0,255,255), -1)
        cv2.imshow("Capturing", frame)
        
        key = cv2.waitKey(1)
        if key == ord('s'):
            nb_files = len(list(Path(".").glob("saved_img_*.jpg")))
            cv2.imwrite(filename=f"saved_img_{nb_files}.jpg", img=frame)
            webcam.release()
            img_new = cv2.imread(f"saved_img_{nb_files}.jpg", cv2.IMREAD_GRAYSCALE)
            img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            print("Processing image...")
            # img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
            # print("Converting RGB image to grayscale...")
            # gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            # print("Converted RGB image to grayscale...")
            # print("Resizing image to 28x28 scale...")
            # img_ = cv2.resize(gray,(28,28))
            # print("Resized...")
            # img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
            # print("Image saved!")
        
            break
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break