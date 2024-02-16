from flask import Flask, request, jsonify

from inference import preprocess_and_predict
from run_measurement import get_measurements

app = Flask(__name__)


@app.route('/predict', methods=["POST"])
def send_prediction():

    try:
        # Get the image url from the request
        image_url = request.args.get('image_url')

        # Perform inference using the TensorFlow model
        keypoints = preprocess_and_predict(image_url)

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
