import cv2 as cv
import numpy as np

# run pip install opencv-python in a cmd window before running to install opencv
# you can also find examples online with static images if you don't have a webcam

# will only detect circles in this range (in terms of pixels on the screen)
# also, it is not perfect, but works well on a flat background with strong contrast, which is our use case

MIN_RADIUS = 1
MAX_RADIUS = 100

def main():
    # Start video capture from the webcam
    cap = cv.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error opening video capture")
        return -1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break

        # Convert to grayscale
        # Should be using contrasting colors for ball and plane
        # i.e. black and white, blue and orange
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Apply median blur
        gray = cv.medianBlur(gray, 5)

        # Detect circles using Hough Transform
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, gray.shape[0] / 8,
                                  param1=100, param2=30,
                                  minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS)

        # Draw circles
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]
                # Draw the outer circle
                cv.circle(frame, center, radius, (0, 255, 0), 2)
                # Draw the center of the circle
                # (these coorinates would be sent to the balancing code)
                cv.circle(frame, center, 2, (0, 0, 255), 3)

        # Display the frame with detected circles
        cv.imshow("Detected Circles", frame)

        # Break the loop if 'q' is pressed 
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        # Need this code otherwise window never closes even if you press X
        if cv.getWindowProperty("Detected Circles",cv.WND_PROP_VISIBLE) < 1:        
            break   

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()