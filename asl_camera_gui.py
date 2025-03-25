import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import tensorflow as tf
import collections

class ASLDetectorApp:
    def __init__(self, window, window_title):
        # Initialize parameters for smoothing and confidence
        self.min_confidence = 0.7
        self.prediction_history = collections.deque(maxlen=10)
        self.current_prediction = None
        self.current_confidence = 0.0
        
        # Set up the main window
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1000x800")
        
        # Set up the video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Load the model and class mapping
        self.model = self.load_model()
        
        # Create GUI elements
        self.create_widgets()
        
        # Start video processing
        self.process_video()
        
        # Set the window close handler
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def load_model(self):
        """Load the trained ASL classifier model"""
        # Check for different possible model filenames
        model_options = [
            'asl_classifier_finetuned.h5'  # Add the new filename
        ]
        
        model_path = None
        for option in model_options:
            if os.path.exists(option):
                model_path = option
                break
                
        if model_path is None:
            raise FileNotFoundError("No ASL classifier model found. Please train the model first.")
            
        print(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        
        # Load class mapping
        self.class_mapping = np.load('class_mapping.npy', allow_pickle=True).item()
        print("Loaded class mapping:", self.class_mapping)
        
        return model
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame (video display)
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Video display label
        self.video_label = ttk.Label(left_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Right frame (controls and results)
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)
        right_frame.pack_propagate(False)
        
        # Title
        title_label = ttk.Label(right_frame, text="ASL Sign Detection", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Instructions
        instructions_frame = ttk.LabelFrame(right_frame, text="Instructions")
        instructions_frame.pack(fill=tk.X, padx=5, pady=5)
        
        instructions_text = (
            "1. Position your hand in the center of the frame\n"
            "2. Make an ASL sign with your hand\n"
            "3. Hold the sign steady for best results\n"
            "4. Ensure good lighting for accuracy"
        )
        instructions_label = ttk.Label(instructions_frame, text=instructions_text, justify=tk.LEFT)
        instructions_label.pack(padx=5, pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(right_frame, text="Detection Results")
        results_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Detected sign
        sign_frame = ttk.Frame(results_frame)
        sign_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(sign_frame, text="Detected Sign:").pack(side=tk.LEFT)
        self.sign_label = ttk.Label(sign_frame, text="None", font=("Arial", 24, "bold"))
        self.sign_label.pack(side=tk.RIGHT)
        
        # Confidence
        conf_frame = ttk.Frame(results_frame)
        conf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(conf_frame, text="Confidence:").pack(side=tk.LEFT)
        self.conf_label = ttk.Label(conf_frame, text="0%")
        self.conf_label.pack(side=tk.RIGHT)
        
        # Confidence meter
        ttk.Label(results_frame, text="Confidence Level:").pack(anchor=tk.W, padx=5)
        self.conf_meter = ttk.Progressbar(results_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.conf_meter.pack(fill=tk.X, padx=5, pady=5)
        
        # Confidence threshold slider
        threshold_frame = ttk.LabelFrame(right_frame, text="Settings")
        threshold_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(threshold_frame, text="Confidence Threshold:").pack(anchor=tk.W, padx=5)
        self.threshold_slider = ttk.Scale(threshold_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, value=self.min_confidence)
        self.threshold_slider.pack(fill=tk.X, padx=5, pady=5)
        
        self.threshold_value_label = ttk.Label(threshold_frame, text=f"{int(self.min_confidence*100)}%")
        self.threshold_value_label.pack(pady=5)
        
        # Bind the slider to update the threshold value
        self.threshold_slider.bind("<Motion>", self.update_threshold)
    
    def update_threshold(self, event=None):
        """Update the confidence threshold based on slider value"""
        self.min_confidence = self.threshold_slider.get()
        self.threshold_value_label.config(text=f"{int(self.min_confidence*100)}%")
    
    def preprocess_frame(self, frame):
        """Preprocess the frame for the model"""
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # Define skin color range in YCrCb
        lower_skin = np.array([0, 135, 85], dtype=np.uint8)
        upper_skin = np.array([255, 180, 135], dtype=np.uint8)
        
        # Create binary mask for skin detection
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Apply the mask to the original frame
        skin = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Check if there's enough hand content in the frame
        if np.sum(mask) / (mask.shape[0] * mask.shape[1] * 255) < 0.01:
            return None
        
        # Convert to RGB (for model input)
        rgb_frame = cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)
        
        # Resize to the input size expected by the model
        resized = cv2.resize(rgb_frame, (224, 224))
        
        # Normalize pixel values
        normalized = resized / 255.0
        
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    
    def get_smoothed_prediction(self, prediction, confidence):
        """Apply smoothing to predictions to reduce flickering"""
        # Add current prediction to history
        self.prediction_history.append((prediction, confidence))
        
        # Count occurrences of each prediction
        prediction_counts = {}
        confidence_sums = {}
        
        for pred, conf in self.prediction_history:
            if pred in prediction_counts:
                prediction_counts[pred] += 1
                confidence_sums[pred] += conf
            else:
                prediction_counts[pred] = 1
                confidence_sums[pred] = conf
        
        # Find the most frequent prediction
        max_count = 0
        smoothed_prediction = None
        smoothed_confidence = 0.0
        
        for pred, count in prediction_counts.items():
            if count > max_count:
                max_count = count
                smoothed_prediction = pred
                smoothed_confidence = confidence_sums[pred] / count
        
        return smoothed_prediction, smoothed_confidence
    
    def process_video(self):
        """Process video frames for sign detection"""
        ret, frame = self.cap.read()
        
        if ret:
            # Flip the frame for a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Draw a rectangle in the center to guide hand positioning
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (w//2-150, h//2-150), (w//2+150, h//2+150), (0, 255, 0), 2)
            
            # Extract the center region for processing
            center_frame = frame[h//2-150:h//2+150, w//2-150:w//2+150].copy()
            
            # Preprocess the frame
            processed_frame = self.preprocess_frame(center_frame)
            
            if processed_frame is not None:
                # Make prediction
                predictions = self.model.predict(processed_frame, verbose=0)[0]
                
                # Get top prediction
                top_idx = np.argmax(predictions)
                confidence = predictions[top_idx]
                prediction = self.class_mapping[top_idx]
                
                # Apply smoothing
                smoothed_prediction, smoothed_confidence = self.get_smoothed_prediction(prediction, confidence)
                
                # Update the results if confidence is above threshold
                if smoothed_confidence >= self.min_confidence:
                    self.current_prediction = smoothed_prediction
                    self.current_confidence = smoothed_confidence
                    
                    # Update labels
                    self.sign_label.config(text=self.current_prediction)
                    self.conf_label.config(text=f"{int(self.current_confidence*100)}%")
                    self.conf_meter['value'] = self.current_confidence * 100
                    
                    # Overlay prediction on frame
                    text = f"{self.current_prediction}: {int(self.current_confidence*100)}%"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convert frame to PhotoImage for display
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = ImageTk.PhotoImage(image=img)
            
            # Store a reference to prevent garbage collection
            self.current_photo = img
            
            # Update video display
            self.video_label.configure(image=self.current_photo)
        
        # Schedule the next frame processing
        self.window.after(10, self.process_video)
    
    def on_close(self):
        """Handle window close event"""
        self.cap.release()
        self.window.destroy()

def main():
    # Create the main window
    root = tk.Tk()
    app = ASLDetectorApp(root, "ASL Sign Detector")
    root.mainloop()

if __name__ == "__main__":
    main() 