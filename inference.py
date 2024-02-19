import tensorflow as tf
import tensorflow_hub as hub

# Model url
model_url = 'https://tfhub.dev/google/movenet/singlepose/thunder/4'

# Global variable to store the loaded model
loaded_model = None


def load_model(model_url):
    global loaded_model
    if loaded_model is None:
        # Load the model if not already loaded
        loaded_model = hub.load(model_url).signatures["serving_default"]


def preprocess_and_predict(image_content, model_url=model_url):
    try:
        # Load the model (or reuse the loaded model)
        load_model(model_url)

        # Set the model input size
        input_size = 256

        # Decode image from content
        input_image = tf.image.decode_image(image_content, channels=3)

        # Expand dimensions, resize, and cast
        input_image = tf.expand_dims(input_image, axis=0)
        input_image = tf.image.resize_with_pad(
            input_image, input_size, input_size)
        input_image = tf.cast(input_image, dtype=tf.int32)

        # Perform inference on the model
        outputs = loaded_model(input_image)
        keypoints_with_scores = outputs["output_0"].numpy()

        # Extract keypoints x,y
        keypoints = keypoints_with_scores[..., :2]

        # Return outputs as keypoints with scores
        return keypoints.tolist()  # Convert to list for JSON serialization
    except ValueError as ve:
        return {'error': str(ve)}, 400  # Bad Request
    except Exception as e:
        # Internal Server Error
        return {'error': f'model inference error, {str(e)}'}, 500
