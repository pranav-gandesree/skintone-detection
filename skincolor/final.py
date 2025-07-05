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
        
        # Define skin tone categories with RGB ranges - Updated for better accuracy
        self.skin_tone_categories = {
            'Very Fair': {
                'rgb_range': {'R': (220, 255), 'G': (200, 240), 'B': (180, 220)},
                'description': 'Very light, pale skin tones'
            },
            'Fair': {
                'rgb_range': {'R': (190, 220), 'G': (160, 200), 'B': (140, 180)},
                'description': 'Light, peachy, pink, or soft beige tones'
            },
            'Light Medium': {
                'rgb_range': {'R': (160, 190), 'G': (130, 160), 'B': (100, 140)},
                'description': 'Light brown, beige, olive tones'
            },
            'Medium': {
                'rgb_range': {'R': (130, 160), 'G': (100, 130), 'B': (70, 100)},
                'description': 'Medium brown, warm beige'
            },
            'Medium Dark': {
                'rgb_range': {'R': (100, 130), 'G': (70, 100), 'B': (50, 70)},
                'description': 'Warm tan, golden brown'
            },
            'Dark': {
                'rgb_range': {'R': (70, 100), 'G': (50, 70), 'B': (30, 50)},
                'description': 'Deep brown tones'
            },
            'Very Dark': {
                'rgb_range': {'R': (40, 70), 'G': (30, 50), 'B': (20, 30)},
                'description': 'Very deep brown to ebony tones'
            }
        }
        
        # Face mesh landmark indices for different face regions
        self.region_landmarks = {
            'forehead': [10, 151, 9, 8, 107, 55, 8, 9, 151, 10],
            'left_cheek': [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147],
            'right_cheek': [345, 346, 347, 348, 349, 350, 355, 371, 266, 425, 426, 427, 436, 416, 376],
            'chin': [18, 175, 199, 200, 9, 10, 151, 175, 18]
        }

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
        
        # Remove extreme outliers (too dark or too bright) - more conservative filtering
        brightness = np.mean(pixels_reshaped, axis=1)
        q10, q90 = np.percentile(brightness, [10, 90])
        
        # Filter out very dark shadows and very bright highlights
        valid_mask = (brightness >= q10) & (brightness <= q90)
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
        confidence = max(0, 1 - (best_score / 400))  # Normalize by max possible distance
        
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
        
        # Work directly with the original image - NO LIGHTING CORRECTION
        working_image = image.copy()
        
        # Get face landmarks
        landmarks = self.get_face_landmarks(working_image)
        if landmarks is None:
            raise ValueError("No face detected in the image")
        
        # Extract and process each region
        region_colors_rgb = {}
        
        for region_name in self.region_landmarks.keys():
            # Extract pixels from region
            region_pixels = self.extract_region_pixels(working_image, landmarks, region_name)
            
            # Apply preprocessing and K-Means
            dominant_rgb = self.preprocess_and_kmeans(region_pixels)
            
            region_colors_rgb[region_name] = dominant_rgb
        
        # Calculate weighted average of regions (give more weight to cheeks)
        weights = {
            'forehead': 0.2,
            'left_cheek': 0.35,
            'right_cheek': 0.35,
            'chin': 0.1
        }
        
        weighted_rgb = np.zeros(3)
        for region_name, rgb in region_colors_rgb.items():
            weighted_rgb += rgb * weights[region_name]
        
        final_rgb = weighted_rgb
        
        # Classify skin tone
        skin_tone_category, confidence = self.classify_skin_tone(final_rgb)
        
        # Prepare results
        results = {
            'skin_tone_category': skin_tone_category,
            'confidence': confidence,
            'final_rgb': final_rgb,
            'region_colors_rgb': region_colors_rgb,
            'category_description': self.skin_tone_categories[skin_tone_category]['description']
        }
        
        if visualize:
            self.visualize_results(image, working_image, landmarks, results)
        
        return results

    def visualize_results(self, original_image: np.ndarray, working_image: np.ndarray, 
                         landmarks, results: Dict):
        """Visualize the detection results."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Working image with landmarks (same as original now)
        working_rgb = cv2.cvtColor(working_image, cv2.COLOR_BGR2RGB)
        axes[0, 1].imshow(working_rgb)
        
        # Draw landmarks for each region
        height, width = working_image.shape[:2]
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
        
        axes[0, 1].set_title('Original Image + Regions')
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
        axes[1, 1].text(0.1, 0.8, f"Category: {results['skin_tone_category']}", 
                       fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f"Confidence: {results['confidence']:.2f}", 
                       fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f"Description: {results['category_description']}", 
                       fontsize=10, wrap=True, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.2, f"RGB Values: R={results['final_rgb'][0]:.0f}, "
                                  f"G={results['final_rgb'][1]:.0f}, B={results['final_rgb'][2]:.0f}", 
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
            
            y_pos -= 0.12
        
        axes[1, 2].set_title('Skin Tone Categories')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize the detector
    detector = SkinToneDetector()

    try:
        image_path = "hema3.jpg" 
        results = detector.detect_skin_tone(image_path, visualize=True)
        
        print("\n=== SKIN TONE DETECTION RESULTS ===")
        print(f"Detected Category: {results['skin_tone_category']}")
        print(f"Confidence: {results['confidence']:.2f}")
        print(f"Final RGB: {results['final_rgb'].astype(int)}")
        print(f"Description: {results['category_description']}")
        
        print("\nRegion Colors (RGB):")
        for region, rgb in results['region_colors_rgb'].items():
            print(f"  {region}: {rgb.astype(int)}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure to provide a valid image path with a clear face visible.")