import sys
import cv2 as cv
import numpy as np

# these are pixels on the screen btw
# will only detect acircles that have a radius in this range
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
                cv.circle(frame, center, 2, (0, 0, 255), 3)

        # Display the frame with detected circles
        cv.imshow("Detected Circles", frame)

        # Break the loop if 'q' is pressed or the window is closed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()