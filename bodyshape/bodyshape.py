import cv2
import mediapipe as mp
import numpy as np
import math

class BodyShapeDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_body_measurements(self, landmarks, image_shape):
        """Extract key body measurements from pose landmarks"""
        h, w = image_shape[:2]
        
        # Get landmark coordinates
        left_shoulder = [landmarks[11].x * w, landmarks[11].y * h]
        right_shoulder = [landmarks[12].x * w, landmarks[12].y * h]
        left_hip = [landmarks[23].x * w, landmarks[23].y * h]
        right_hip = [landmarks[24].x * w, landmarks[24].y * h]
        
        # Calculate measurements
        shoulder_width = self.calculate_distance(left_shoulder, right_shoulder)
        hip_width = self.calculate_distance(left_hip, right_hip)
        
        # Calculate waist width (approximate - midpoint between shoulders and hips)
        waist_left_x = (left_shoulder[0] + left_hip[0]) / 2
        waist_right_x = (right_shoulder[0] + right_hip[0]) / 2
        waist_width = abs(waist_right_x - waist_left_x)
        
        # Get torso length
        shoulder_center = [(left_shoulder[0] + right_shoulder[0]) / 2, 
                          (left_shoulder[1] + right_shoulder[1]) / 2]
        hip_center = [(left_hip[0] + right_hip[0]) / 2, 
                     (left_hip[1] + right_hip[1]) / 2]
        torso_length = self.calculate_distance(shoulder_center, hip_center)
        
        return {
            'shoulder_width': shoulder_width,
            'waist_width': waist_width,
            'hip_width': hip_width,
            'torso_length': torso_length
        }
    
    def classify_body_shape(self, measurements):
        """Classify body shape based on measurements"""
        shoulder_width = measurements['shoulder_width']
        waist_width = measurements['waist_width']
        hip_width = measurements['hip_width']
        
        # Calculate ratios
        shoulder_hip_ratio = shoulder_width / hip_width if hip_width > 0 else 0
        waist_hip_ratio = waist_width / hip_width if hip_width > 0 else 0
        waist_shoulder_ratio = waist_width / shoulder_width if shoulder_width > 0 else 0
        
        # Classification thresholds
        # These can be adjusted based on your specific requirements
        
        # Hourglass: waist significantly smaller than both shoulders and hips
        if (waist_shoulder_ratio < 0.75 and waist_hip_ratio < 0.75 and 
            abs(shoulder_hip_ratio - 1.0) < 0.15):
            return "Hourglass"
        
        # Inverted Triangle: shoulders significantly wider than hips
        elif shoulder_hip_ratio > 1.2:
            return "Inverted Triangle"
        
        # Pear: hips significantly wider than shoulders
        elif shoulder_hip_ratio < 0.85:
            return "Pear"
        
        # Apple: waist similar to or larger than hips, shoulders wider than hips
        elif (waist_hip_ratio > 0.85 and shoulder_hip_ratio > 1.0):
            return "Apple"
        
        # Rectangle: measurements are relatively similar
        else:
            return "Rectangle"
    
    def detect_body_shape(self, image_path):
        """Main function to detect body shape from image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None, "Error: Could not load image"
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(rgb_image)
        
        if not results.pose_landmarks:
            return None, "No pose landmarks detected"
        
        # Get body measurements
        measurements = self.get_body_measurements(results.pose_landmarks.landmark, image.shape)
        
        # Classify body shape
        body_shape = self.classify_body_shape(measurements)
        
        # Draw pose landmarks on image
        annotated_image = image.copy()
        self.mp_drawing.draw_landmarks(
            annotated_image, 
            results.pose_landmarks, 
            self.mp_pose.POSE_CONNECTIONS
        )
        
        # Add text with body shape and measurements
        cv2.putText(annotated_image, f"Body Shape: {body_shape}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add measurements text
        y_offset = 70
        for key, value in measurements.items():
            text = f"{key}: {value:.1f}px"
            cv2.putText(annotated_image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
        
        return annotated_image, body_shape, measurements
    
    def resize_image_for_display(self, image, max_width=600, max_height=500):
        """Resize image to fit within specified dimensions while maintaining aspect ratio"""
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h)
        
        # Always resize to fit within window
        new_width = int(w * scale)
        new_height = int(h * scale)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_image, scale
    
    def process_image_with_display(self, image_path):
        """Process image and display results"""
        result = self.detect_body_shape(image_path)
        
        if result[0] is None:
            print(result[1])
            return
        
        annotated_image, body_shape, measurements = result
        
        print(f"Detected Body Shape: {body_shape}")
        print("\nMeasurements:")
        for key, value in measurements.items():
            print(f"  {key}: {value:.1f}px")
        
        # Resize image for display
        display_image, scale_factor = self.resize_image_for_display(annotated_image)
        
        # Adjust font size based on scale factor
        font_scale = max(0.4, min(0.8, scale_factor))
        thickness = max(1, int(2 * scale_factor))
        
        # Re-add text with appropriate scaling
        cv2.putText(display_image, f"Body Shape: {body_shape}", 
                   (10, int(25 * font_scale * 2)), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (0, 255, 0), thickness)
        
        # Add measurements text with scaling
        y_offset = int(50 * font_scale * 2)
        for key, value in measurements.items():
            text = f"{key}: {value:.1f}px"
            cv2.putText(display_image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, 
                       (255, 255, 255), max(1, thickness - 1))
            y_offset += int(20 * font_scale * 2)
        
        # Display the resized image with window properties
        cv2.namedWindow('Body Shape Detection', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Body Shape Detection', display_image)
        
        print(f"\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save the original annotated image (full size)
        output_path = f"output_{body_shape.lower().replace(' ', '_')}.jpg"
        cv2.imwrite(output_path, annotated_image)
        print(f"\nAnnotated image saved as: {output_path}")
        print(f"Display scale factor: {scale_factor:.2f}")

# Usage example
if __name__ == "__main__":
    # Initialize detector
    detector = BodyShapeDetector()
    
    # Example usage - replace with your image path
    image_path = "img69.jpg"  # Replace with your image path
    
    try:
        detector.process_image_with_display(image_path)
    except Exception as e:
        print(f"Error processing image: {e}")
        print("Make sure:")
        print("1. The image path is correct")
        print("2. The image contains a visible person")
        print("3. MediaPipe and OpenCV are properly installed")
    
    # Alternative: Process without display (useful for batch processing)
    """
    result = detector.detect_body_shape(image_path)
    if result[0] is not None:
        annotated_image, body_shape, measurements = result
        print(f"Body Shape: {body_shape}")
        cv2.imwrite("output.jpg", annotated_image)
    """