import cv2 as cv
import mediapipe as mp
import numpy as np
from sklearn.cluster import KMeans

# Skin tone categories
# SKIN_TONE_CATEGORIES = {
#     "Fair":   {"r": (210, 255), "g": (180, 230), "b": (170, 220)},
#     "Medium": {"r": (170, 210), "g": (130, 180), "b": (110, 160)},
#     "Tan":    {"r": (130, 170), "g": (100, 140), "b": (80, 130)},
#     "Dark":   {"r": (60, 130),  "g": (50, 100),  "b": (40, 90)},
# }

SKIN_TONE_CATEGORIES = {
    "Fair":   {"r": (190, 255), "g": (170, 240), "b": (160, 235)},
    "Medium": {"r": (150, 190), "g": (120, 180), "b": (100, 170)},
    "Tan":    {"r": (110, 150), "g": (90, 140),  "b": (70, 130)},
    "Dark":   {"r": (50, 110),  "g": (40, 100),  "b": (30, 90)},
}




# Lighting correction: Gray-world white balance
def gray_world_white_balance(img):
    img = img.astype(np.float32)
    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])
    avg_gray = (avg_r + avg_g + avg_b) / 3
    img[:, :, 0] *= (avg_gray / avg_b)
    img[:, :, 1] *= (avg_gray / avg_g)
    img[:, :, 2] *= (avg_gray / avg_r)
    return np.clip(img, 0, 255).astype(np.uint8)

# Preprocess skin patch
def preprocess_patch(patch, size=(20, 20)):
    patch = cv.resize(patch, size)
    gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
    mask = cv.inRange(gray, 40, 220)
    flat = patch.reshape(-1, 3)
    return flat[mask.flatten() > 0]

# K-Means clustering
def find_dominant_color(pixels, k=3):
    if len(pixels) < k:
        return None
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(pixels)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant = kmeans.cluster_centers_[labels[np.argmax(counts)]]
    return dominant.astype(int)

# Classify skin tone
def classify_skin_tone(rgb):
    r, g, b = rgb
    for tone, bounds in SKIN_TONE_CATEGORIES.items():
        if bounds["r"][0] <= r <= bounds["r"][1] and bounds["g"][0] <= g <= bounds["g"][1] and bounds["b"][0] <= b <= bounds["b"][1]:
            return tone
    return "Unknown"

# Face Mesh landmark groups
# REGION_LANDMARKS = {
#     "forehead": [10, 338, 297],
#     "left_cheek": [234, 93, 132],
#     "right_cheek": [454, 323, 361],
#     "chin": [152, 200, 210],
# }

REGION_LANDMARKS = {
    "forehead": [10, 338, 297],
    "left_cheek": [93, 205, 187],       # new closer center
    "right_cheek": [323, 425, 411],     # new closer center
    "chin": [152, 200, 210],
}


# Load and correct lighting
img = cv.imread('image.jpg')
if img is None:
    print("Image not found.")
    exit()
img = gray_world_white_balance(img)

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = img.shape

        for region_name, lm_indices in REGION_LANDMARKS.items():
            pts = []
            for idx in lm_indices:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                pts.append((x, y))
            cx = sum(p[0] for p in pts) // len(pts)
            cy = sum(p[1] for p in pts) // len(pts)

            PATCH_SIZE = 30
            x1, y1 = max(0, cx - PATCH_SIZE // 2), max(0, cy - PATCH_SIZE // 2)
            x2, y2 = x1 + PATCH_SIZE, y1 + PATCH_SIZE
            patch = img[y1:y2, x1:x2]

            # Preprocess and classify
            flat_pixels = preprocess_patch(patch)
            dominant_color = find_dominant_color(flat_pixels)

            if dominant_color is not None:
                skin_tone = classify_skin_tone(dominant_color)
                print(f"{region_name}: RGB={dominant_color.tolist()} → Tone={skin_tone}")

                # Show dominant color
                color_patch = np.zeros((50, 50, 3), dtype=np.uint8)
                color_patch[:] = dominant_color[::-1]  # RGB to BGR
                cv.imshow(f"{region_name}_color", color_patch)

            # Visual debug
            cv.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

    cv.imshow("Face Mesh with Skin Patches", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
