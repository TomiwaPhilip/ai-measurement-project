from flask import Flask, request, jsonify
import secrets
import requests

from inference import preprocess_and_predict
from run_measurement import get_measurements

app = Flask(__name__)

app.secret_key = secrets.token_hex(16)


@app.route('/predict', methods=["POST"])
def send_prediction():
    try:
        # Get the image URL from the request
        image_url = request.form.get('image_url')

        # Download the image from the URL
        try:
            response_image = requests.get(image_url)
            response_image.raise_for_status()  # Check for any download errors
            image_content = response_image.content
        except requests.exceptions.RequestException as e:
            return jsonify({'error': f'Error downloading image: {str(e)}'}), 400


        # Perform inference using the TensorFlow model
        keypoints = preprocess_and_predict(image_content)

        # Perform measurements
        results = get_measurements(keypoints)

        # Return the result
        return jsonify({'result': results})

    except Exception as e:
        # Handle exceptions, log them, or return an error response
        error_message = f"An error occurred: {str(e)}"
        # HTTP status code 500 for Internal Server Error
        return jsonify({'error': error_message}), 500


if __name__ == '__main__':
    app.run(debug=True)
