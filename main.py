from flask import Flask, request
import jsonify

from inference import preprocess_and_predict
from run_measurement import get_measurements

app = Flask(__name__)


@app.route('/predict', methods=["POST"])
def send_prediction():

    # Get the image from the request
    image_data = request.get_data()

    # Perform inference using the TensorFlow model
    keypoints = preprocess_and_predict(image_data)

    # Perform measurements
    results = get_measurements(keypoints)

    # Return the result
    return jsonify({'result': results})


if __name__ == '__main__':
    app.run(debug=True)
