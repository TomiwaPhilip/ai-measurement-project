from flask import Flask, request

app = Flask(__name__)

@app.route('/model-prediction', methods=["POST"])
def process_image():
 
    # Get the image from the request
    image_data = request.get_data()

    # Process the image (e.g., perform any necessary decoding or resizing)
    processed_image = preprocess_image(image_data)

    # Perform inference using the TensorFlow model
    result = your_model.predict(processed_image)

    # Return the result
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
