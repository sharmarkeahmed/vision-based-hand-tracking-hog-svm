import cv2
import time
import numpy as np

def apply_kmeans(frame, K=8):
    # Convert the frame to a 2D float32 array of pixels
    Z = frame.reshape((-1, 3))
    Z = np.float32(Z)

    # K-means criteria and run
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Convert centers back to uint8 and rebuild segmented image
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented = segmented.reshape(frame.shape)
    return segmented

# def _get_FPS():
    

def main():
    # """
    # Start the camera frame with
    # Args: 
    #     cam: choice of camera in the system (defult 0)

    # Return:

    # """

    TARGET_FPS = 20
    FRAME_INTERVAL = 1.0 / TARGET_FPS
    K = 4   # Number of clusters for K-means

    cap = cv2.VideoCapture(cam)

    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    # print(f"Running with K-means (K={K}) at ~{TARGET_FPS} FPS")

    last_time = 0
    fps_last_time = time.time()
    fps_counter = 0
    current_fps = 0

    while True:
        now = time.time()

        if now - last_time >= FRAME_INTERVAL:
            last_time = now
            ret, frame = cap.read()
            if not ret:
                break

            # Apply K-means color segmentation
            segmented_frame = apply_kmeans(frame, K)

            # FPS calculation
            fps_counter += 1
            if now - fps_last_time >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_last_time = now

            # Overlay FPS text
            cv2.putText(segmented_frame, f"FPS: {current_fps}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            # Show segmented image
            cv2.imshow("K-means Camera", segmented_frame)

        # Quit on q
        if cv2.waitKey(1) & 0xFF == 'q':
            break

    cap.release()
    cv2.destroyAllWindows()


# def main():
#     while True:



if __name__ == "__main__":
    main()
