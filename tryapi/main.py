from flask import Flask,request,jsonify
import numpy as np
import pickle



model = pickle.load(open('trylastend.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"
@app.route('/predict',methods=['POST'])
def predict():
    query = request.form.get('query')
    input_query = np.array([query])
    result = model.predict(input_query)
    # print(result)
    return jsonify({'Answer':str(result)})
if __name__ == '__main__':
    app.run(debug=True)