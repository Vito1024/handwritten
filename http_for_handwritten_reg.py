from flask import Flask, jsonify, request, render_template
import torch


app = Flask(__name__, template_folder='./')

@app.route("/")
def index():
    return render_template('handwritten_reg.html')

@app.route("/reg", methods=['POST'])
def reg():
    
    json = request.get_json()

    bs = json['bs']
    img = torch.tensor(bs).reshape(28, 28)

    response = jsonify({'image': img.shape})
    return response


if __name__ == '__main__':
    app.run(debug=True)
