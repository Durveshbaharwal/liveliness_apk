import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="liveliness.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Model input shape
input_shape = input_details[0]['shape']
height = int(input_shape[1] / 1.5)
width = input_shape[2]

def convert_bgr_to_yuv420(image):
    """
    Convert BGR image (from OpenCV) to YUV420 planar format with single channel.
    Final shape: (H * 1.5, W, 1)
    """
    image_resized = cv2.resize(image, (width, height))
    yuv_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2YUV_I420)
    yuv_image = np.reshape(yuv_image, (int(height * 1.5), width, 1))
    return yuv_image.astype(np.uint8)

def predict_liveness(frame):
    input_tensor = convert_bgr_to_yuv420(frame)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    return int(output)  # Output: 0 = live, 1 = spoof, 2 = multiple_faces

# Label mapping
label_map = {
    0: "Live",
    1: "Spoof",
    2: "Multiple Faces"
}

# OpenCV capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    try:
        result = predict_liveness(frame)
        label = label_map.get(result, "Unknown")
    except Exception as e:
        label = f"Error: {str(e)}"

    # Display result on frame
    display_frame = cv2.resize(frame, (width, height))
    cv2.putText(display_frame, f"Liveness: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if label == "Live" else (0, 0, 255), 2)
    cv2.imshow("Liveness Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
