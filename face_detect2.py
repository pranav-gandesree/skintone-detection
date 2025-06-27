# import cv2 as cv
# import mediapipe as mp
# import numpy as np
# from sklearn.cluster import KMeans
# from skimage import color

# # --- Updated Skin Tone Category Bounds in CIE Lab Space ---
# # CIE Lab color space gives better perception-based thresholds (L: 0-100, a/b: -128 to +127)
# SKIN_TONE_CATEGORIES_LAB = {
#     "Fair":   {"L": (65, 100), "a": (-5, 20), "b": (10, 30)},
#     "Medium": {"L": (55, 70),  "a": (0, 25),  "b": (15, 35)},
#     "Tan":    {"L": (40, 60),  "a": (5, 30),  "b": (20, 40)},
#     "Dark":   {"L": (20, 45),  "a": (0, 25),  "b": (15, 35)},
# }


# # --- Lighting Correction ---
# def gray_world_white_balance(img):
#     img = img.astype(np.float32)
#     avg_b, avg_g, avg_r = [np.mean(img[:, :, i]) for i in range(3)]
#     avg_gray = (avg_r + avg_g + avg_b) / 3
#     for i, avg in enumerate([avg_b, avg_g, avg_r]):
#         img[:, :, i] *= (avg_gray / avg)
#     return np.clip(img, 0, 255).astype(np.uint8)

# # --- Patch Preprocessing ---
# def preprocess_patch(patch, size=(20, 20)):
#     patch = cv.resize(patch, size)
#     gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
#     mask = cv.inRange(gray, 40, 220)
#     flat = patch.reshape(-1, 3)
#     return flat[mask.flatten() > 0]

# # --- KMeans + Dominant Color ---
# def find_dominant_color(pixels, k=3):
#     if len(pixels) < k:
#         return None
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
#     kmeans.fit(pixels)
#     labels, counts = np.unique(kmeans.labels_, return_counts=True)
#     dominant = kmeans.cluster_centers_[labels[np.argmax(counts)]]
#     return dominant.astype(int)

# # --- Lab Color Classification ---
# def classify_lab_color(lab):
#     L, a, b = lab
#     for tone, ranges in SKIN_TONE_CATEGORIES_LAB.items():
#         if ranges["L"][0] <= L <= ranges["L"][1] and ranges["a"][0] <= a <= ranges["a"][1] and ranges["b"][0] <= b <= ranges["b"][1]:
#             return tone
#     return "Unknown"

# # --- Landmark Groups for Patches ---
# REGION_LANDMARKS = {
#     "forehead": [10, 338, 297],
#     "left_cheek": [93, 205, 187],
#     "right_cheek": [323, 425, 411],
#     "chin": [152, 200, 210],
# }

# # --- Load and Correct Image ---
# img = cv.imread('image.jpg')
# if img is None:
#     print("Image not found.")
#     exit()

# img = gray_world_white_balance(img)

# # --- Face Mesh Detection ---
# mp_face_mesh = mp.solutions.face_mesh
# with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
#     rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_img)

#     if results.multi_face_landmarks:
#         face_landmarks = results.multi_face_landmarks[0]
#         h, w, _ = img.shape

#         lab_list = []

#         for region_name, lm_indices in REGION_LANDMARKS.items():
#             pts = []
#             for idx in lm_indices:
#                 lm = face_landmarks.landmark[idx]
#                 x, y = int(lm.x * w), int(lm.y * h)
#                 pts.append((x, y))
#             cx = sum(p[0] for p in pts) // len(pts)
#             cy = sum(p[1] for p in pts) // len(pts)

#             PATCH_SIZE = 30
#             x1, y1 = max(0, cx - PATCH_SIZE // 2), max(0, cy - PATCH_SIZE // 2)
#             x2, y2 = x1 + PATCH_SIZE, y1 + PATCH_SIZE
#             patch = img[y1:y2, x1:x2]

#             flat_pixels = preprocess_patch(patch)
#             dom_rgb = find_dominant_color(flat_pixels)
#             if dom_rgb is None:
#                 continue

#             # Convert RGB → Lab (scikit-image expects range [0, 1])
#             rgb_norm = np.array(dom_rgb[::-1]) / 255.0  # BGR → RGB → normalize
#             lab_color = color.rgb2lab([[rgb_norm]])[0][0]
#             lab_list.append(lab_color)

#             # Debug output
#             color_patch = np.zeros((40, 40, 3), dtype=np.uint8)
#             color_patch[:] = dom_rgb[::-1]
#             cv.imshow(f"{region_name}_color", color_patch)

#             # Draw patch
#             cv.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

#         # --- Final Average Skin Tone ---
#         if lab_list:
#             avg_lab = np.mean(lab_list, axis=0)
#             skin_tone = classify_lab_color(avg_lab)
#             print(f"Average Lab: {avg_lab} → Skin Tone: {skin_tone}")

#             # Show final color and label
#             avg_rgb = color.lab2rgb([[avg_lab]])[0][0] * 255
#             avg_rgb = avg_rgb.astype(np.uint8)

#             color_patch = np.zeros((50, 100, 3), dtype=np.uint8)
#             color_patch[:] = avg_rgb[::-1]  # RGB to BGR
#             cv.imshow(f"Avg Tone: {skin_tone}", color_patch)


#             # Show label
#             cv.putText(img, f"Skin Tone: {skin_tone}", (30, 40), cv.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
#         else:
#             print("No valid skin regions found.")

#     cv.imshow("Final Detection", img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

































import cv2 as cv
import mediapipe as mp
import numpy as np
from sklearn.cluster import KMeans
from skimage import color

# --- Skin Tone Categories in CIE Lab Space ---
SKIN_TONE_CATEGORIES_LAB = {
    "Fair":   {"L": (65, 100), "a": (-5, 20), "b": (10, 30)},
    "Medium": {"L": (55, 70),  "a": (0, 25),  "b": (15, 35)},
    "Tan":    {"L": (40, 60),  "a": (5, 30),  "b": (20, 40)},
    "Dark":   {"L": (20, 45),  "a": (0, 25),  "b": (15, 35)},
}

def gray_world_white_balance(img):
    img = img.astype(np.float32)
    avg_b, avg_g, avg_r = [np.mean(img[:, :, i]) for i in range(3)]
    avg_gray = (avg_r + avg_g + avg_b) / 3
    for i, avg in enumerate([avg_b, avg_g, avg_r]):
        img[:, :, i] *= (avg_gray / avg)
    return np.clip(img, 0, 255).astype(np.uint8)

def preprocess_patch(patch, size=(20, 20)):
    patch = cv.resize(patch, size)
    gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
    mask = cv.inRange(gray, 40, 220)
    flat = patch.reshape(-1, 3)
    return flat[mask.flatten() > 0]

def find_dominant_color(pixels, k=3):
    if len(pixels) < k:
        return None
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(pixels)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant = kmeans.cluster_centers_[labels[np.argmax(counts)]]
    return dominant.astype(int)

def classify_lab_color(lab):
    L, a, b = lab
    for tone, ranges in SKIN_TONE_CATEGORIES_LAB.items():
        if ranges["L"][0] <= L <= ranges["L"][1] and ranges["a"][0] <= a <= ranges["a"][1] and ranges["b"][0] <= b <= ranges["b"][1]:
            return tone
    return "Unknown"

REGION_LANDMARKS = {
    "forehead": [10, 338, 297],
    "left_cheek": [93, 205, 187],
    "right_cheek": [323, 425, 411],
    "chin": [152, 200, 210],
}

img = cv.imread('image.jpg')
if img is None:
    print("Image not found.")
    exit()

img = gray_world_white_balance(img)

mp_face_mesh = mp.solutions.face_mesh
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = img.shape

        lab_list = []

        for region_name, lm_indices in REGION_LANDMARKS.items():
            pts = [(int(face_landmarks.landmark[idx].x * w),
                    int(face_landmarks.landmark[idx].y * h)) for idx in lm_indices]
            cx = sum(p[0] for p in pts) // len(pts)
            cy = sum(p[1] for p in pts) // len(pts)

            PATCH_SIZE = 30
            x1, y1 = max(0, cx - PATCH_SIZE // 2), max(0, cy - PATCH_SIZE // 2)
            x2, y2 = x1 + PATCH_SIZE, y1 + PATCH_SIZE
            patch = img[y1:y2, x1:x2]

            flat_pixels = preprocess_patch(patch)
            dom_rgb = find_dominant_color(flat_pixels)
            if dom_rgb is None:
                continue

            rgb_norm = np.array(dom_rgb[::-1]) / 255.0
            lab_color = color.rgb2lab([[rgb_norm]])[0][0]
            lab_list.append(lab_color)

            # Draw rectangle on face
            cv.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

        if lab_list:
            avg_lab = np.mean(lab_list, axis=0)
            skin_tone = classify_lab_color(avg_lab)
            print(f"Average Lab: {avg_lab} → Skin Tone: {skin_tone}")

            cv.putText(img, f"Skin Tone: {skin_tone}", (30, 40),
                       cv.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
        else:
            print("No valid skin regions found.")

    # ✅ Only show main image
    cv.imshow("Final Detection", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
