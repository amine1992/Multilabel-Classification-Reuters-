from flask import Flask, request, jsonify
from keras.models import load_model

app = Flask(__name__)


@app.route("/reuters", methods=['POST'])
def analyse_reuters():
    doc = request.get_json()['doc']
    # TODO: implement heurestic to reject 
    rejected = False
    prediction = []

    if not rejected:
    	model = load_model('mlp_reuters.h5')
    	prediction = model.predict(doc)
    	
    return jsonify(
        Class=prediction,
        Rejected=rejected
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
