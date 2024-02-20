from flask import Flask, request, jsonify
import secrets
import requests
import tempfile
import os

from inference import preprocess_and_predict
from run_measurement import get_measurements

app = Flask(__name__)

app.secret_key = secrets.token_hex(16)


def get_file_extension(content_type):
    # Determine file extension based on content type
    if 'jpeg' in content_type:
        return 'jpg'
    elif 'png' in content_type:
        return 'png'
    else:
        return 'raw'


@app.route('/predict', methods=["POST"])
def send_prediction():
    try:

        # Get the image URL from the request
        image_url = request.form.get('image_url')

        # Download the image from the URL
        try:
            response_image = requests.get(image_url)
            response_image.raise_for_status()  # Check for any download errors

            # Determine file extension based on content type
            content_type = response_image.headers.get('content-type')
            file_extension = get_file_extension(content_type)

            # Save the image to a temporary file with the appropriate extension
            with tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False) as temp_file:
                temp_file.write(response_image.content)
                temp_file_path = temp_file.name

        except requests.exceptions.RequestException as e:
            return jsonify({'error': f'Error downloading image: {str(e)}'}), 400

        # Perform inference using the TensorFlow model
        keypoints = preprocess_and_predict(temp_file_path)

        # Perform measurements
        results = get_measurements(keypoints)

        # Delete the temporary file
        os.remove(temp_file_path)

        # Return the result
        return jsonify(results), 200, {'Content-Type': 'application/json', 'sort_keys': False}

    except Exception as e:
        # Handle exceptions, log them, or return an error response
        error_message = f"An error occurred: {str(e)}"
        # HTTP status code 500 for Internal Server Error
        return jsonify({'error': error_message}), 500


if __name__ == '__main__':
    app.run(debug=True)
