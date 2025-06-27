# # This code reads an image from the file 'image.jpg' and displays it in a window.
# # If the image cannot be read, it prints an error message.
# # Make sure to have an image named 'image.jpg' in the same directory as this script.
# # You can change the filename to any valid image file you have.

# import cv2 as cv
# import mediapipe as mp
# import numpy as np

# # Load image
# img = cv.imread('image.jpg')
# if img is None:
#     print("Error: Image not found.")
#     exit()

# # Initialize MediaPipe Face Detection
# mp_face_detection = mp.solutions.face_detection
# with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector:
#     # Convert BGR to RGB
#     rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#     # Process the image
#     results = face_detector.process(rgb_img)

#     frame_height, frame_width, _ = img.shape

#     if results.detections:
#         for face in results.detections:
#             # Get bounding box
#             face_rect = np.multiply(
#                 [
#                     face.location_data.relative_bounding_box.xmin,
#                     face.location_data.relative_bounding_box.ymin,
#                     face.location_data.relative_bounding_box.width,
#                     face.location_data.relative_bounding_box.height,
#                 ],
#                 [frame_width, frame_height, frame_width, frame_height]
#             ).astype(int)

#             x, y, w, h = face_rect
#             cv.rectangle(img, (x, y), (x + w, y + h), color=(255, 255, 255), thickness=2)

#             # Draw key points
#             key_points = np.array([(p.x, p.y) for p in face.location_data.relative_keypoints])
#             key_points_coords = np.multiply(key_points, [frame_width, frame_height]).astype(int)
#             for p in key_points_coords:
#                 cv.circle(img, tuple(p), 4, (255, 255, 255), 2)
#                 cv.circle(img, tuple(p), 2, (0, 0, 0), -1)

#     # Show the result
#     cv.imshow("Face Detection", img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
























import cv2 as cv
import mediapipe as mp
import numpy as np

# Load image
img = cv.imread('image.jpg')
if img is None:
    print("Error: Image not found.")
    exit()

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector:
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = face_detector.process(rgb_img)
    frame_height, frame_width, _ = img.shape

    if results.detections:
        for face in results.detections:
            bbox = face.location_data.relative_bounding_box
            x, y, w, h = np.multiply(
                [bbox.xmin, bbox.ymin, bbox.width, bbox.height],
                [frame_width, frame_height, frame_width, frame_height]
            ).astype(int)

            # Get keypoints
            keypoints = face.location_data.relative_keypoints
            kp = lambda i: np.array([int(keypoints[i].x * frame_width), int(keypoints[i].y * frame_height)])

            left_eye = kp(0)
            right_eye = kp(1)
            nose_tip = kp(2)
            mouth_center = kp(3)

            # Approximate face center
            face_cx = x + w // 2
            face_cy = y + h // 2

            # Define skin region patches (30x30 px around each center)
            PATCH_SIZE = 30

            # Cheeks
            left_cheek_center = (left_eye[0] - 40, left_eye[1] + 30)
            right_cheek_center = (right_eye[0] + 40, right_eye[1] + 30)

            # Forehead (above midpoint between eyes)
            forehead_center = ((left_eye[0] + right_eye[0]) // 2, left_eye[1] - 50)

            # Chin (below mouth)
            chin_center = (mouth_center[0], mouth_center[1] + 40)

            regions = {
                "left_cheek": left_cheek_center,
                "right_cheek": right_cheek_center,
                "forehead": forehead_center,
                "chin": chin_center
            }

            for name, (cx, cy) in regions.items():
                x1, y1 = max(0, cx - PATCH_SIZE // 2), max(0, cy - PATCH_SIZE // 2)
                x2, y2 = x1 + PATCH_SIZE, y1 + PATCH_SIZE
                patch = img[y1:y2, x1:x2]

                # Save or show each region
                cv.imshow(name, patch)
                cv.imwrite(f"{name}.jpg", patch)  # Optional: save to file

            # Draw bounding box and label
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

    cv.imshow("Detected Face", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
