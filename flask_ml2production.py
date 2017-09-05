# flask_ml2production.py
import numpy as np
from flask import Flask, abort, jsonify,request
import pickle

my_random_forest = pickle.load(open('iris_rfc.pkl', 'rb'))

app = Flask(__name__)

def make_predict():
	data = request.get_json(force=True)
	print(data)
	predict_request = [data['sl'], data['sw'], data['pl'], data['pw']]
	predict_request = np.array(predict_request).reshape((1, -1))
	y_hat = my_random_forest.predict(predict_request)
	print(y_hat)
	output = str(y_hat[0])
	return jsonify(results=output)


@app.route('/')
def index():
 return "This is ML API"

@app.route('/api', methods=["POST"])
def api_predict():
 return make_predict()	


if __name__ == '__main__':
	app.run(port=9000, debug=True)