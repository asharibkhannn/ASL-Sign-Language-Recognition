import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import time
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import argparse
from collections import defaultdict

def load_asl_model():
    """Load the trained ASL classifier model"""
    model_options = ['asl_classifier_finetuned.h5', 'asl_classifier_best.h5', 'asl_classifier_final.h5']
    
    model_path = None
    for option in model_options:
        if os.path.exists(option):
            model_path = option
            break
            
    if model_path is None:
        raise FileNotFoundError("No ASL classifier model found.")
        
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    # Load class mapping
    class_mapping = np.load('class_mapping.npy', allow_pickle=True).item()
    print("Loaded class mapping:", class_mapping)
    
    # Invert class mapping for easier use
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    return model, idx_to_class

def preprocess_image(image_path):
    """Preprocess a single image for prediction"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
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
    skin = cv2.bitwise_and(img, img, mask=mask)
    
    # Check if there's enough hand content in the frame
    if np.sum(mask) / (mask.shape[0] * mask.shape[1] * 255) < 0.01:
        return None
    
    # Convert to RGB (for model input)
    rgb_img = cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)
    
    # Resize to the input size expected by the model
    resized = cv2.resize(rgb_img, (224, 224))
    
    # Normalize pixel values
    normalized = resized / 255.0
    
    return normalized

def predict_single_image(model, image, idx_to_class):
    """Predict the ASL sign in a single preprocessed image"""
    # Add batch dimension
    input_img = np.expand_dims(image, axis=0)
    
    # Time the prediction
    start_time = time.time()
    predictions = model.predict(input_img, verbose=0)[0]
    inference_time = time.time() - start_time
    
    # Get top prediction
    top_idx = np.argmax(predictions)
    confidence = predictions[top_idx]
    prediction = idx_to_class[top_idx]
    
    return prediction, confidence, inference_time

def evaluate_test_data(model, test_dir, idx_to_class):
    """Evaluate model on test data and return metrics"""
    y_true = []
    y_pred = []
    confidences = []
    inference_times = []
    class_accuracies = defaultdict(list)
    
    # Get all subdirectories in test_dir (each subdirectory represents a class)
    classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    
    total_images = 0
    processed_images = 0
    
    # Count total number of images for progress bar
    for class_name in classes:
        class_dir = os.path.join(test_dir, class_name)
        total_images += len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"Evaluating {total_images} test images...")
    progress_bar = tqdm(total=total_images, unit="images")
    
    for class_name in classes:
        class_dir = os.path.join(test_dir, class_name)
        class_images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        correct_for_class = 0
        total_for_class = 0
        
        for image_file in class_images:
            image_path = os.path.join(class_dir, image_file)
            processed_image = preprocess_image(image_path)
            
            if processed_image is not None:
                prediction, confidence, inference_time = predict_single_image(model, processed_image, idx_to_class)
                
                y_true.append(class_name)
                y_pred.append(prediction)
                confidences.append(confidence)
                inference_times.append(inference_time)
                processed_images += 1
                
                # Track per-class accuracy
                total_for_class += 1
                if prediction == class_name:
                    correct_for_class += 1
                
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Calculate overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Calculate per-class accuracy
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        class_accuracies[true].append(1 if true == pred else 0)
    
    per_class_accuracy = {class_name: sum(vals)/len(vals) for class_name, vals in class_accuracies.items()}
    
    # Calculate average confidence and inference time
    avg_confidence = np.mean(confidences)
    avg_inference_time = np.mean(inference_times)
    
    conf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    class_report = classification_report(y_true, y_pred, labels=classes)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'per_class_accuracy': per_class_accuracy,
        'avg_confidence': avg_confidence,
        'avg_inference_time': avg_inference_time,
        'conf_matrix': conf_matrix,
        'class_report': class_report,
        'classes': classes,
        'y_true': y_true,
        'y_pred': y_pred,
        'confidences': confidences,
        'processed_images': processed_images,
        'total_images': total_images
    }
    
    return metrics

def plot_confusion_matrix(conf_matrix, classes, save_path="confusion_matrix.png"):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(14, 12))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved confusion matrix to {save_path}")
    plt.close()

def plot_per_class_accuracy(per_class_accuracy, save_path="per_class_accuracy.png"):
    """Plot and save per-class accuracy"""
    plt.figure(figsize=(12, 8))
    classes = list(per_class_accuracy.keys())
    accuracies = list(per_class_accuracy.values())
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_accuracies = [accuracies[i] for i in sorted_indices]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_classes)))
    
    bars = plt.bar(sorted_classes, sorted_accuracies, color=colors)
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Threshold')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', rotation=0)
    
    plt.xlabel('ASL Class')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per ASL Class')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved per-class accuracy to {save_path}")
    plt.close()

def plot_confidence_distribution(confidences, save_path="confidence_distribution.png"):
    """Plot and save confidence score distribution"""
    plt.figure(figsize=(10, 6))
    sns.histplot(confidences, bins=20, kde=True)
    plt.axvline(x=0.7, color='r', linestyle='--', label='Threshold (0.7)')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Confidence Scores')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved confidence distribution to {save_path}")
    plt.close()

def plot_timing_analysis(inference_times, save_path="inference_times.png"):
    """Plot and save inference time analysis"""
    plt.figure(figsize=(10, 6))
    sns.histplot(inference_times, bins=20, kde=True)
    plt.axvline(x=np.mean(inference_times), color='r', linestyle='--', 
                label=f'Mean: {np.mean(inference_times):.4f}s')
    plt.xlabel('Inference Time (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Inference Times')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved inference time analysis to {save_path}")
    plt.close()

def compare_preprocessing_methods(model, test_image_path, idx_to_class):
    """Compare different preprocessing methods on a single test image"""
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return
    
    original_img = cv2.imread(test_image_path)
    if original_img is None:
        print(f"Failed to load image: {test_image_path}")
        return
    
    # Method 1: Standard YCrCb skin segmentation (current method)
    def preprocess_ycrcb(img):
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 135, 85], dtype=np.uint8)
        upper_skin = np.array([255, 180, 135], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        skin = cv2.bitwise_and(img, img, mask=mask)
        rgb_img = cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)
        return rgb_img
    
    # Method 2: HSV skin segmentation
    def preprocess_hsv(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
        lower_skin = np.array([170, 20, 70], dtype=np.uint8)
        upper_skin = np.array([180, 255, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = mask1 + mask2
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        skin = cv2.bitwise_and(img, img, mask=mask)
        rgb_img = cv2.cvtColor(skin, cv2.COLOR_BGR2RGB)
        return rgb_img
    
    # Method 3: Simple grayscale
    def preprocess_gray(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray_3c = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return gray_3c
    
    # Method 4: CLAHE enhancement
    def preprocess_clahe(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
        enhanced_3c = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return enhanced_3c
    
    # Apply preprocessing methods
    methods = {
        "YCrCb Skin Segmentation": preprocess_ycrcb,
        "HSV Skin Segmentation": preprocess_hsv,
        "Grayscale": preprocess_gray,
        "CLAHE Enhancement": preprocess_clahe
    }
    
    results = {}
    
    plt.figure(figsize=(15, 12))
    
    # Add original image
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    for i, (name, preprocess_func) in enumerate(methods.items(), 2):
        processed = preprocess_func(original_img.copy())
        
        # Resize for model
        resized = cv2.resize(processed, (224, 224))
        normalized = resized / 255.0
        
        # Predict
        prediction, confidence, inference_time = predict_single_image(model, normalized, idx_to_class)
        
        results[name] = {
            "prediction": prediction,
            "confidence": confidence,
            "inference_time": inference_time
        }
        
        # Display
        plt.subplot(2, 3, i)
        plt.imshow(processed)
        plt.title(f"{name}\nPrediction: {prediction} ({confidence:.2f})")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("preprocessing_comparison.png")
    print("Saved preprocessing method comparison to preprocessing_comparison.png")
    plt.close()
    
    return results

def analyze_thresholds(y_true, y_pred, confidences, save_path="threshold_analysis.png"):
    """Analyze impact of different confidence thresholds"""
    thresholds = np.linspace(0, 1, 21)
    accuracies = []
    kept_samples = []
    
    for threshold in thresholds:
        # Keep only predictions with confidence >= threshold
        keep_indices = [i for i, conf in enumerate(confidences) if conf >= threshold]
        
        if len(keep_indices) > 0:
            kept_true = [y_true[i] for i in keep_indices]
            kept_pred = [y_pred[i] for i in keep_indices]
            
            # Calculate accuracy on kept samples
            acc = accuracy_score(kept_true, kept_pred)
            kept = len(keep_indices) / len(confidences)
            
            accuracies.append(acc)
            kept_samples.append(kept)
        else:
            accuracies.append(None)
            kept_samples.append(0)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, 'b-', label='Accuracy')
    plt.plot(thresholds, kept_samples, 'r--', label='Fraction of Samples Kept')
    plt.axvline(x=0.7, color='g', linestyle=':', label='Default Threshold (0.7)')
    
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Value')
    plt.title('Effect of Confidence Threshold on Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved threshold analysis to {save_path}")
    plt.close()

def create_performance_report(metrics, output_file="performance_report.md"):
    """Generate a markdown performance report"""
    report = f"""# ASL Sign Language Recognition - Performance Report

## Overall Performance Metrics

- **Accuracy**: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
- **Precision**: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)
- **Recall**: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)
- **F1 Score**: {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)
- **Average Confidence**: {metrics['avg_confidence']:.4f} ({metrics['avg_confidence']*100:.2f}%)
- **Average Inference Time**: {metrics['avg_inference_time']*1000:.2f} ms per image
- **Images Processed**: {metrics['processed_images']} out of {metrics['total_images']} ({metrics['processed_images']/metrics['total_images']*100:.2f}%)

## Classification Report

```
{metrics['class_report']}
```

## Top and Bottom Performing Classes

### Top 5 Classes:
"""
    
    # Add top 5 performing classes
    sorted_classes = sorted(metrics['per_class_accuracy'].items(), key=lambda x: x[1], reverse=True)
    for class_name, accuracy in sorted_classes[:5]:
        report += f"- **{class_name}**: {accuracy:.4f} ({accuracy*100:.2f}%)\n"
    
    report += "\n### Bottom 5 Classes:\n"
    
    # Add bottom 5 performing classes
    for class_name, accuracy in sorted_classes[-5:]:
        report += f"- **{class_name}**: {accuracy:.4f} ({accuracy*100:.2f}%)\n"
    
    report += """
## Performance Visualizations

The following visualizations have been generated:

1. **Confusion Matrix**: Shows the number of correct and incorrect predictions for each class
2. **Per-Class Accuracy**: Displays the accuracy for each ASL sign
3. **Confidence Distribution**: Shows the distribution of confidence scores
4. **Inference Times**: Shows the distribution of inference times
5. **Threshold Analysis**: Shows how accuracy changes with different confidence thresholds
6. **Preprocessing Comparison**: Compares different preprocessing methods

## Recommendations

Based on the performance metrics, here are some recommendations:

1. Focus on improving recognition for the bottom 5 performing classes
2. Consider adjusting the confidence threshold (current default: 0.7)
3. Evaluate preprocessing methods for specific challenging signs
4. Collect additional training data for underperforming classes
"""

    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"Performance report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate ASL Sign Language Recognition model performance')
    parser.add_argument('--test_dir', type=str, help='Directory containing test images organized by class')
    parser.add_argument('--test_image', type=str, help='Single test image for preprocessing comparison', default=None)
    args = parser.parse_args()
    
    # Create output directory for results
    output_dir = "performance_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and class mapping
    model, idx_to_class = load_asl_model()
    
    if args.test_dir:
        # Evaluate model on test data
        metrics = evaluate_test_data(model, args.test_dir, idx_to_class)
        
        # Plot confusion matrix
        plot_confusion_matrix(metrics['conf_matrix'], metrics['classes'], 
                             save_path=os.path.join(output_dir, "confusion_matrix.png"))
        
        # Plot per-class accuracy
        plot_per_class_accuracy(metrics['per_class_accuracy'], 
                               save_path=os.path.join(output_dir, "per_class_accuracy.png"))
        
        # Plot confidence distribution
        plot_confidence_distribution(metrics['confidences'], 
                                    save_path=os.path.join(output_dir, "confidence_distribution.png"))
        
        # Plot timing analysis
        plot_timing_analysis(metrics['inference_times'], 
                            save_path=os.path.join(output_dir, "inference_times.png"))
        
        # Analyze thresholds
        analyze_thresholds(metrics['y_true'], metrics['y_pred'], metrics['confidences'], 
                          save_path=os.path.join(output_dir, "threshold_analysis.png"))
        
        # Create performance report
        create_performance_report(metrics, output_file=os.path.join(output_dir, "performance_report.md"))
    
    if args.test_image:
        # Compare preprocessing methods on a single image
        compare_preprocessing_methods(model, args.test_image, idx_to_class)
    
    if not args.test_dir and not args.test_image:
        print("Error: Please specify either --test_dir or --test_image")
        print("Example usage:")
        print("  python evaluate_model_performance.py --test_dir path/to/test_data")
        print("  python evaluate_model_performance.py --test_image path/to/image.jpg")
        print("  python evaluate_model_performance.py --test_dir path/to/test_data --test_image path/to/image.jpg")

if __name__ == "__main__":
    main() 