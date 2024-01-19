from flask import Flask, jsonify, request, render_template
import torch

from LeNet import LeNet, device


model = LeNet()
model.load_state_dict(
    torch.load('./handwritten.model', map_location=torch.device(device))
)
model.eval()

app = Flask(__name__, template_folder='./')

@app.route("/")
def index():
    return render_template('handwritten_reg.html')

@app.route("/reg", methods=['POST'])
def reg():
    
    json = request.get_json()
    bs = json['bs']

    img = torch.tensor(bs, dtype=torch.float).reshape(1, 1, 28, 28)
    evaluated = model(img).argmax(dim=1).item()

    response = jsonify({'evaluated': evaluated})
    return response


if __name__ == '__main__':
    app.run(debug=True)
