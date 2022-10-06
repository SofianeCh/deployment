import os
import logging
import socket
import torch
from flask import Flask, jsonify, request


HOST_NAME = os.environ.get('APP_DNS', 'localhost')
APP_NAME = os.environ.get('APP_NAME', 'flask')
IP = os.environ.get('PYTHON_IP', '127.0.0.1')
PORT = int(os.environ.get('PYTHON_PORT', 8080))
HOME_DIR = os.environ.get('HOMEDIR', os.getcwd())

log = logging.getLogger(__name__)
app = Flask(__name__)


@app.route('/')
def hello():
    return jsonify({
        'host_name': HOST_NAME,
        'app_name': APP_NAME,
        'ip': IP,
        'port': PORT,
        'home_dir': HOME_DIR,
        'host': socket.gethostname()
    })


def get_model():
    # This will change once the model is hosted on hugginsface.
    path = "seq2seqmodel"
    model = torch.load(path)
    return model


@app.route('/prediction', methods=['POST'])
def get_prediction():
    try:
        model = get_model()
        if request.method == 'POST':
            to_analize = request.json.get('text')
            result = model.predict([to_analize])
            return jsonify(result), 200
    except Exception as e:
        print(e)
        return jsonify({"error": e}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)
