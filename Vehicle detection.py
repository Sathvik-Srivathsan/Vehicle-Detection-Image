import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# --- Configuration ---
# URL for a pre-trained SSD MobileNet V2 FPNLite model from TensorFlow Hub.
# This model is trained on the COCO dataset, which includes 'car', 'truck', 'bus', 'motorcycle'.
# The model expects input images of size 320x320 pixels.
MODEL_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"

# Confidence threshold: Only display detections with a score above this value.
CONFIDENCE_THRESHOLD = 0.5

# List of COCO dataset class names that we consider 'vehicles'.
# The COCO dataset has 90 classes, and these are the relevant ones for vehicle detection.
# Note: The class IDs from the model are 1-indexed for COCO, so we map them correctly.
# We'll use a full COCO labels mapping to ensure correct class names.
COCO_LABELS = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'mixer', 82: 'sink',
    83: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
    88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}
# Filter for vehicle-related classes based on their names
VEHICLE_CLASS_NAMES = ['car', 'truck', 'bus', 'motorcycle']
VEHICLE_CLASS_IDS = [k for k, v in COCO_LABELS.items() if v in VEHICLE_CLASS_NAMES]


def load_object_detection_model(model_url):
    """
    Loads a pre-trained object detection model from TensorFlow Hub.

    Args:
        model_url (str): The URL of the model on TensorFlow Hub.

    Returns:
        tf.Module: The loaded TensorFlow model.
    """
    print(f"Loading model from: {model_url}...")
    try:
        # hub.load downloads and loads the model into memory.
        # It handles the graph construction and weight loading.
        model = hub.load(model_url)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an active internet connection and the URL is correct.")
        exit() # Exit the program if the model cannot be loaded


def preprocess_image(image_path, target_size=(320, 320)):
    """
    Loads an image, converts it to RGB, resizes it, and prepares it for model input.

    Args:
        image_path (str): The file path to the image.
        target_size (tuple): The (width, height) the model expects.

    Returns:
        tuple: A tuple containing:
            - original_image (np.array): The original image loaded by OpenCV (BGR format).
            - input_tensor (tf.Tensor): The preprocessed image ready for model inference.
    """
    # Read the image using OpenCV. cv2.imread loads images in BGR format by default.
    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Error: Could not load image from {image_path}. Please check the path.")
        return None, None

    # Convert BGR image (OpenCV default) to RGB (TensorFlow models usually expect RGB).
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Convert the NumPy array to a TensorFlow tensor.
    # The model expects uint8 type.
    input_tensor = tf.convert_to_tensor(rgb_image, dtype=tf.uint8)

    # Add a batch dimension: The model expects input in the shape [batch_size, height, width, channels].
    # For a single image, batch_size is 1.
    input_tensor = tf.expand_dims(input_tensor, 0)

    # Resize the image to the model's expected input size.
    # tf.image.resize expects float32, so we cast, resize, then cast back to uint8.
    input_tensor = tf.image.resize(input_tensor, target_size)
    input_tensor = tf.cast(input_tensor, dtype=tf.uint8) # Cast back if model expects uint8

    print(f"Image '{os.path.basename(image_path)}' loaded and preprocessed.")
    return original_image, input_tensor


def perform_detection(model, input_tensor):
    """
    Performs object detection using the loaded model.
    Handles potential variations in model output structure for 'num_detections'.

    Args:
        model (tf.Module): The loaded TensorFlow object detection model.
        input_tensor (tf.Tensor): The preprocessed image tensor.

    Returns:
        dict: A dictionary containing detection results (boxes, scores, classes, num_detections).
    """
    print("Performing object detection...")
    # Run inference. The model returns a dictionary of detection results.
    # The output tensors are typically tf.Tensor objects.
    detections_raw = model(input_tensor)

    # Convert all relevant tensors to NumPy arrays for easier handling.
    # The [0] selects the first item from the batch (since we only have one image).
    # .numpy() converts the TensorFlow tensor to a NumPy array.
    processed_detections = {
        'detection_boxes': detections_raw['detection_boxes'][0].numpy(),
        'detection_scores': detections_raw['detection_scores'][0].numpy(),
        'detection_classes': detections_raw['detection_classes'][0].numpy()
        # Other keys like 'raw_detection_boxes', 'raw_detection_scores', etc.,
        # are not directly used for drawing but can be included if needed.
    }

    # Get the actual number of valid detections.
    # Instead of relying on a 'num_detections' key that might not exist,
    # we use the length of the 'detection_scores' array. This array is always
    # truncated by the model to only contain valid detections.
    num_valid_detections = processed_detections['detection_scores'].shape[0]

    # detection_classes are usually float64, convert to int for indexing labels
    processed_detections['detection_classes'] = processed_detections['detection_classes'].astype(np.int64)

    # Add 'num_detections' key to the processed dictionary.
    # This makes the output consistent for the 'draw_detections' function.
    processed_detections['num_detections'] = num_valid_detections

    print(f"Found {num_valid_detections} actual detections (based on scores).")
    return processed_detections


def draw_detections(image, detections, confidence_threshold, class_ids, class_labels):
    """
    Draws bounding boxes and labels on the image for detected objects.

    Args:
        image (np.array): The original image (OpenCV BGR format) to draw on.
        detections (dict): Dictionary of detection results.
        confidence_threshold (float): Minimum confidence score to display a detection.
        class_ids (list): List of class IDs to filter for (e.g., vehicle IDs).
        class_labels (dict): Mapping from class ID to human-readable label.

    Returns:
        np.array: The image with bounding boxes and labels drawn.
    """
    # Get image dimensions
    img_height, img_width, _ = image.shape

    # Iterate through detections
    num_deteions_drawn = 0
    for i in range(detections['num_detections']):
        score = detections['detection_scores'][i]
        class_id = detections['detection_classes'][i]

        # Check if the detection meets the confidence threshold and is a vehicle class
        if score >= confidence_threshold and class_id in class_ids:
            num_deteions_drawn += 1
            # Bounding box coordinates are normalized [0, 1] and in [ymin, xmin, ymax, xmax] format
            ymin, xmin, ymax, xmax = detections['detection_boxes'][i]

            # Convert normalized coordinates to pixel coordinates
            (left, right, top, bottom) = (xmin * img_width, xmax * img_width,
                                          ymin * img_height, ymax * img_height)

            # Cast to integers for drawing with OpenCV
            left, right, top, bottom = int(left), int(right), int(top), int(bottom)

            # Draw the bounding box (rectangle)
            # cv2.rectangle(image, start_point, end_point, color, thickness)
            # Color is BGR (Blue, Green, Red) tuple
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2) # Red color, 2px thickness

            # Get the class name and format the label text
            class_name = class_labels.get(class_id, 'Unknown')
            label = f"{class_name}: {score:.2f}"

            # Put text label on the image
            # cv2.putText(image, text, org, fontFace, fontScale, color, thickness, lineType)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = left
            text_y = top - 10 if top - 10 > 10 else top + text_size[1] + 10 # Adjust y for visibility

            # Draw a filled rectangle as background for the text for better readability
            cv2.rectangle(image, (text_x, text_y - text_size[1] - 5),
                          (text_x + text_size[0] + 5, text_y + 5), (0, 0, 255), -1) # Red background

            # Put the text on the image
            cv2.putText(image, label, (text_x + 2, text_y - 2), font, font_scale,
                        (255, 255, 255), font_thickness, cv2.LINE_AA) # White text

    print(f"Drawn {num_deteions_drawn} vehicle detections.")
    return image


def main():
    """
    Main function to run the vehicle detection program.
    """
    # Load the pre-trained object detection model
    detector = load_object_detection_model(MODEL_URL)

    while True:
        # Prompt user for image path
        image_path = input("\nEnter the path to the image file (or 'q' to quit): ").strip()

        if image_path.lower() == 'q':
            print("Exiting program.")
            break

        if not os.path.exists(image_path):
            print(f"Error: File not found at '{image_path}'. Please enter a valid path.")
            continue

        # Preprocess the image for the model
        original_image, input_tensor = preprocess_image(image_path)
        if original_image is None: # Error during preprocessing
            continue

        # Perform detection
        detections = perform_detection(detector, input_tensor)

        # Draw bounding boxes and labels on the original image
        output_image = draw_detections(original_image.copy(), detections,
                                       CONFIDENCE_THRESHOLD, VEHICLE_CLASS_IDS, COCO_LABELS)

        # Display the image with detections using matplotlib
        # OpenCV's imshow can sometimes be problematic in PyCharm/certain environments
        # Matplotlib is generally more robust for displaying images in scripts.
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for matplotlib
        plt.title(f"Vehicle Detections in {os.path.basename(image_path)}")
        plt.axis('off') # Hide axes
        plt.show()

        print("\nDetection complete. Close the image window to continue or quit.")

if __name__ == "__main__":
    # Ensure TensorFlow only logs errors to keep console clean
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()

