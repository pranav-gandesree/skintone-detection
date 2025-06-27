import cv2
import numpy as np
import mediapipe as mp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional

class SkinToneDetector:
    def __init__(self):
        """Initialize MediaPipe Face Mesh and skin tone categories."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7
        )
        
        # Define skin tone categories with RGB ranges
        self.skin_tone_categories = {
            'Fair': {
                'rgb_range': {'R': (210, 255), 'G': (180, 230), 'B': (170, 220)},
                'description': 'Light, peachy, pink, or soft beige tones'
            },
            'Medium': {
                'rgb_range': {'R': (170, 210), 'G': (130, 180), 'B': (110, 160)},
                'description': 'Light brown, beige, olive tones'
            },
            'Tan': {
                'rgb_range': {'R': (130, 170), 'G': (100, 140), 'B': (80, 130)},
                'description': 'Warm tan, golden brown'
            },
            'Dark': {
                'rgb_range': {'R': (60, 130), 'G': (50, 100), 'B': (40, 90)},
                'description': 'Rich brown to deep ebony tones'
            }
        }
        
        # Face mesh landmark indices for different face regions
        self.region_landmarks = {
            'forehead': [10, 151, 9, 8, 107, 55, 8, 9, 151, 10],
            'left_cheek': [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147],
            'right_cheek': [345, 346, 347, 348, 349, 350, 355, 371, 266, 425, 426, 427, 436, 416, 376],
            'chin': [18, 175, 199, 200, 9, 10, 151, 175, 18]
        }

    def apply_gray_world_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply Gray World lighting correction to the image."""
        # Convert to float for processing
        image_float = image.astype(np.float32)
        
        # Calculate mean for each channel
        mean_r = np.mean(image_float[:, :, 2])  # Red channel
        mean_g = np.mean(image_float[:, :, 1])  # Green channel
        mean_b = np.mean(image_float[:, :, 0])  # Blue channel
        
        # Calculate gray world assumption scaling factors
        gray_mean = (mean_r + mean_g + mean_b) / 3
        
        scale_r = gray_mean / mean_r if mean_r > 0 else 1
        scale_g = gray_mean / mean_g if mean_g > 0 else 1
        scale_b = gray_mean / mean_b if mean_b > 0 else 1
        
        # Apply correction
        corrected = image_float.copy()
        corrected[:, :, 2] = np.clip(corrected[:, :, 2] * scale_r, 0, 255)  # Red
        corrected[:, :, 1] = np.clip(corrected[:, :, 1] * scale_g, 0, 255)  # Green
        corrected[:, :, 0] = np.clip(corrected[:, :, 0] * scale_b, 0, 255)  # Blue
        
        return corrected.astype(np.uint8)

    def get_face_landmarks(self, image: np.ndarray) -> Optional[List]:
        """Extract face landmarks using MediaPipe Face Mesh."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None

    def extract_region_pixels(self, image: np.ndarray, landmarks, region_name: str) -> np.ndarray:
        """Extract pixels from a specific face region using landmarks."""
        height, width = image.shape[:2]
        landmark_indices = self.region_landmarks[region_name]
        
        # Convert normalized landmarks to pixel coordinates
        points = []
        for idx in landmark_indices:
            x = int(landmarks.landmark[idx].x * width)
            y = int(landmarks.landmark[idx].y * height)
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        
        # Create mask for the region
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        
        # Extract pixels from the masked region
        region_pixels = image[mask > 0]
        return region_pixels

    def preprocess_and_kmeans(self, pixels: np.ndarray, n_clusters: int = 3) -> np.ndarray:
        """Preprocess pixels and apply K-Means clustering to find dominant color."""
        if len(pixels) == 0:
            return np.array([0, 0, 0])
        
        # Reshape pixels for clustering
        pixels_reshaped = pixels.reshape(-1, 3)
        
        # Remove extreme outliers (too dark or too bright)
        brightness = np.mean(pixels_reshaped, axis=1)
        q1, q3 = np.percentile(brightness, [25, 75])
        iqr = q3 - q1
        lower_bound = max(0, q1 - 1.5 * iqr)
        upper_bound = min(255, q3 + 1.5 * iqr)
        
        valid_mask = (brightness >= lower_bound) & (brightness <= upper_bound)
        filtered_pixels = pixels_reshaped[valid_mask]
        
        if len(filtered_pixels) < n_clusters:
            return np.mean(pixels_reshaped, axis=0)
        
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(filtered_pixels)
        
        # Get the cluster centers and their sizes
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # Find the largest cluster (most representative color)
        unique_labels, counts = np.unique(labels, return_counts=True)
        dominant_cluster_idx = unique_labels[np.argmax(counts)]
        
        return centers[dominant_cluster_idx]

    def rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to CIE Lab color space."""
        # Normalize RGB to [0, 1]
        rgb_normalized = rgb / 255.0
        
        # Convert to XYZ color space first
        def gamma_correction(channel):
            return np.where(channel > 0.04045, 
                          np.power((channel + 0.055) / 1.055, 2.4),
                          channel / 12.92)
        
        rgb_corrected = gamma_correction(rgb_normalized)
        
        # RGB to XYZ transformation matrix (sRGB to XYZ)
        transformation_matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        xyz = np.dot(transformation_matrix, rgb_corrected)
        
        # Normalize by D65 illuminant
        xyz_normalized = xyz / np.array([0.95047, 1.00000, 1.08883])
        
        # XYZ to Lab conversion
        def f_transform(t):
            return np.where(t > 0.008856, 
                          np.power(t, 1/3),
                          (7.787 * t) + (16/116))
        
        fx, fy, fz = f_transform(xyz_normalized)
        
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return np.array([L, a, b])

    def lab_to_rgb(self, lab: np.ndarray) -> np.ndarray:
        """Convert CIE Lab to RGB color space."""
        L, a, b = lab
        
        # Lab to XYZ conversion
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        
        def f_inverse(t):
            return np.where(t > 0.206893, 
                          np.power(t, 3),
                          (t - 16/116) / 7.787)
        
        xyz_normalized = np.array([f_inverse(fx), f_inverse(fy), f_inverse(fz)])
        
        # Denormalize by D65 illuminant
        xyz = xyz_normalized * np.array([0.95047, 1.00000, 1.08883])
        
        # XYZ to RGB transformation matrix
        transformation_matrix = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252]
        ])
        
        rgb_linear = np.dot(transformation_matrix, xyz)
        
        # Gamma correction
        def inverse_gamma_correction(channel):
            return np.where(channel > 0.0031308,
                          1.055 * np.power(channel, 1/2.4) - 0.055,
                          12.92 * channel)
        
        rgb_corrected = inverse_gamma_correction(rgb_linear)
        
        # Clamp to [0, 1] and convert to [0, 255]
        rgb_final = np.clip(rgb_corrected * 255, 0, 255)
        
        return rgb_final

    def classify_skin_tone(self, rgb: np.ndarray) -> Tuple[str, float]:
        """Classify the final RGB color into one of the skin tone categories."""
        r, g, b = rgb
        
        best_category = None
        best_score = float('inf')
        
        for category_name, category_data in self.skin_tone_categories.items():
            rgb_range = category_data['rgb_range']
            
            # Calculate how well the RGB values fit within the category ranges
            r_fit = self._calculate_range_fit(r, rgb_range['R'])
            g_fit = self._calculate_range_fit(g, rgb_range['G'])
            b_fit = self._calculate_range_fit(b, rgb_range['B'])
            
            # Combined score (lower is better)
            combined_score = r_fit + g_fit + b_fit
            
            if combined_score < best_score:
                best_score = combined_score
                best_category = category_name
        
        # Convert score to confidence (0-1, higher is better)
        confidence = max(0, 1 - (best_score / 300))  # Normalize by max possible distance
        
        return best_category, confidence

    def _calculate_range_fit(self, value: float, range_tuple: Tuple[int, int]) -> float:
        """Calculate how well a value fits within a range."""
        min_val, max_val = range_tuple
        
        if min_val <= value <= max_val:
            return 0  # Perfect fit
        elif value < min_val:
            return min_val - value
        else:
            return value - max_val

    def detect_skin_tone(self, image_path: str, visualize: bool = True) -> Dict:
        """Main function to detect skin tone from an image."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Apply lighting correction
        corrected_image = self.apply_gray_world_correction(image)
        
        # Get face landmarks
        landmarks = self.get_face_landmarks(corrected_image)
        if landmarks is None:
            raise ValueError("No face detected in the image")
        
        # Extract and process each region
        region_colors_rgb = {}
        region_colors_lab = {}
        
        for region_name in self.region_landmarks.keys():
            # Extract pixels from region
            region_pixels = self.extract_region_pixels(corrected_image, landmarks, region_name)
            
            # Apply preprocessing and K-Means
            dominant_rgb = self.preprocess_and_kmeans(region_pixels)
            
            # Convert to Lab
            dominant_lab = self.rgb_to_lab(dominant_rgb)
            
            region_colors_rgb[region_name] = dominant_rgb
            region_colors_lab[region_name] = dominant_lab
        
        # Average all Lab values
        all_lab_values = list(region_colors_lab.values())
        averaged_lab = np.mean(all_lab_values, axis=0)
        
        # Convert back to RGB
        final_rgb = self.lab_to_rgb(averaged_lab)
        
        # Classify skin tone
        skin_tone_category, confidence = self.classify_skin_tone(final_rgb)
        
        # Prepare results
        results = {
            'skin_tone_category': skin_tone_category,
            'confidence': confidence,
            'final_rgb': final_rgb,
            'final_lab': averaged_lab,
            'region_colors_rgb': region_colors_rgb,
            'region_colors_lab': region_colors_lab,
            'category_description': self.skin_tone_categories[skin_tone_category]['description']
        }
        
        if visualize:
            self.visualize_results(image, corrected_image, landmarks, results)
        
        return results

    def visualize_results(self, original_image: np.ndarray, corrected_image: np.ndarray, 
                         landmarks, results: Dict):
        """Visualize the detection results."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Corrected image with landmarks
        corrected_rgb = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)
        axes[0, 1].imshow(corrected_rgb)
        
        # Draw landmarks for each region
        height, width = corrected_image.shape[:2]
        colors = {'forehead': 'red', 'left_cheek': 'blue', 'right_cheek': 'green', 'chin': 'orange'}
        
        for region_name, color in colors.items():
            landmark_indices = self.region_landmarks[region_name]
            points = []
            for idx in landmark_indices:
                x = int(landmarks.landmark[idx].x * width)
                y = int(landmarks.landmark[idx].y * height)
                points.append([x, y])
            points = np.array(points)
            axes[0, 1].plot(points[:, 0], points[:, 1], color=color, linewidth=2, alpha=0.7)
        
        axes[0, 1].set_title('Lighting Corrected + Regions')
        axes[0, 1].axis('off')
        
        # Region colors
        region_colors = results['region_colors_rgb']
        color_patches = []
        labels = []
        
        for i, (region_name, rgb) in enumerate(region_colors.items()):
            color_patch = np.ones((50, 100, 3), dtype=np.uint8)
            color_patch[:, :] = rgb.astype(np.uint8)
            color_patches.append(color_patch)
            labels.append(f'{region_name}\nRGB: {rgb.astype(int)}')
        
        combined_patches = np.hstack(color_patches)
        axes[0, 2].imshow(combined_patches)
        axes[0, 2].set_title('Region Dominant Colors')
        axes[0, 2].axis('off')
        
        # Final skin tone color
        final_rgb = results['final_rgb'].astype(np.uint8)
        final_color = np.ones((200, 300, 3), dtype=np.uint8)
        final_color[:, :] = final_rgb
        
        axes[1, 0].imshow(final_color)
        axes[1, 0].set_title(f'Final Skin Tone\nRGB: {final_rgb}')
        axes[1, 0].axis('off')
        
        # Classification result
        axes[1, 1].text(0.1, 0.7, f"Category: {results['skin_tone_category']}", 
                       fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f"Confidence: {results['confidence']:.2f}", 
                       fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.3, f"Description: {results['category_description']}", 
                       fontsize=10, wrap=True, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.1, f"Lab Values: L={results['final_lab'][0]:.1f}, "
                                  f"a={results['final_lab'][1]:.1f}, b={results['final_lab'][2]:.1f}", 
                       fontsize=10, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Classification Results')
        axes[1, 1].axis('off')
        
        # Color comparison with categories
        axes[1, 2].axis('off')
        y_pos = 0.9
        for category_name, category_data in self.skin_tone_categories.items():
            # Create a sample color patch for each category (using middle values)
            rgb_range = category_data['rgb_range']
            sample_rgb = [
                (rgb_range['R'][0] + rgb_range['R'][1]) // 2,
                (rgb_range['G'][0] + rgb_range['G'][1]) // 2,
                (rgb_range['B'][0] + rgb_range['B'][1]) // 2
            ]
            
            # Highlight the detected category
            if category_name == results['skin_tone_category']:
                axes[1, 2].text(0.05, y_pos, f"→ {category_name}", fontsize=12, 
                               fontweight='bold', color='red', transform=axes[1, 2].transAxes)
            else:
                axes[1, 2].text(0.05, y_pos, f"   {category_name}", fontsize=10, 
                               transform=axes[1, 2].transAxes)
            
            y_pos -= 0.2
        
        axes[1, 2].set_title('Skin Tone Categories')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize the detector
    detector = SkinToneDetector()
    
    # Example usage (replace with your image path)
    try:
        image_path = "image.jpg"  # Replace with actual image path
        results = detector.detect_skin_tone(image_path, visualize=True)
        
        print("\n=== SKIN TONE DETECTION RESULTS ===")
        print(f"Detected Category: {results['skin_tone_category']}")
        print(f"Confidence: {results['confidence']:.2f}")
        print(f"Final RGB: {results['final_rgb'].astype(int)}")
        print(f"Final Lab: L={results['final_lab'][0]:.1f}, a={results['final_lab'][1]:.1f}, b={results['final_lab'][2]:.1f}")
        print(f"Description: {results['category_description']}")
        
        print("\nRegion Colors (RGB):")
        for region, rgb in results['region_colors_rgb'].items():
            print(f"  {region}: {rgb.astype(int)}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure to provide a valid image path with a clear face visible.")