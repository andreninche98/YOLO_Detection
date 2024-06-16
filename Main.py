from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import cv2
import multiprocessing
import json
import base64
import yaml
import os
import frame_handler
import Detection



app = Flask(__name__, template_folder='/home/andrea/PycharmProjects/YOLO_Detection/templates')
socketio = SocketIO(app)
manager = multiprocessing.Manager()
return_dict = manager.dict()
capture_processes = []
capturing = False

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config("/home/andrea/PycharmProjects/YOLO_Detection/sources.yaml")
batch_size = config['batch_size']

worker_processes = [
    multiprocessing.Process(target=frame_handler.start_worker, args=("queue_yolo",)),
    multiprocessing.Process(target=frame_handler.start_worker, args=("queue_tiny",)),
    multiprocessing.Process(target=frame_handler.start_worker, args=("queue_mid",)),
]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results_page():
    return render_template('result_page.html')

@app.route('/start', methods=['POST'])
def start_stream():
    global capture_processes, capturing
    capturing = True
    for p in worker_processes:
        p.start()

    for source in config['sources']:
        p = multiprocessing.Process(target=capture_frames, args=(source, return_dict))
        capture_processes.append(p)

    for p in capture_processes:
        p.start()

    return jsonify(status='Detection started')


@app.route('/stop', methods=['POST'])
def stop_stream():
    global capture_processes, capturing
    capturing = False
    for p in capture_processes:
        p.terminate()

    return jsonify(status='Stream stopped')

@app.route('/detect', methods=['POST'])
def detect():
    global return_dict
    for source in config['sources']:
        if source['id'] not in return_dict or len(return_dict[source['id']]) == 0:
            print(f"[WARNING] No frames captured from source: {source}")
        else:
            source_frames = return_dict[source['id']]
            for i in range(0, len(source_frames), batch_size):
                batch = source_frames[i:i + batch_size]
                print(f"[INFO] Processing batch for {source['save_directory']}")
                detect_with_yolo(batch, source['save_directory'])

    return jsonify({"status": "Detection completed"})

@app.route('/status', methods=['GET'])
def status():
    return jsonify(status='Running' if capturing else 'Stopped')

@app.route('/list_saved_frames_metadata', methods=['GET'])
def list_saved_frames_metadata():
    try:
        saved_frames_metadata = load_saved_frames_metadata()
        return jsonify(saved_frames_metadata)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def capture_frames(source, return_dict):
    try:
        frames_list = []
        print(f"[INFO] Starting capture from source: {source['type'], source['id']}")

        if source['type'] == 'webcam':
            cap = cv2.VideoCapture(source['id'])
        else:
            raise ValueError(f"Unsupported source type: {source['type']}")

        if not cap.isOpened():
            raise ValueError(f"Failed to open source: {source['type']}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_list.append(frame)
            print(f"[INFO] Frame captured from source: {source['type'], source['id']}, {len(frames_list)} captured")

            return_dict[source['id']] = frames_list
            emit_frames(source['id'], frame)

        print(f"[INFO] Finished capture from source: {source['type'], source['id']}")
        cap.release()

    except ValueError as ve:
        print(f"[ERROR] {ve}")
    except Exception as e:
        print(f"[ERROR] Error capturing frames: {e}")

def emit_frames(source_id, frame):
    _, buffer = cv2.imencode('.jpg', frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    socketio.emit(f'video_feed_{source_id}', {'source': source_id, 'frame': frame_b64})


def detect_with_yolo(frames, save_path):
    try:
        results = Detection.detect_batch(frames, Detection.models)
        if len(results) >= batch_size:
            frame_handler.add_detection_job(results, save_path)
        return True
    except Exception as e:
        print(f"[ERROR] Error detecting with yolo: {e}")
        return False

def load_saved_frames_metadata():
    metadata = {}
    save_path = "/home/andrea/PycharmProjects/YOLO_Detection/frame_det"

    for root, dirs, files in os.walk(save_path):
        for filename in files:
            if filename.endswith('.json'):
                filepath = os.path.join(root, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)

                    # Esempio di estrazione dei dati dai nomi delle cartelle
                    model_type = os.path.basename(root)
                    date = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
                    class_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(filepath))))

                    if model_type not in metadata:
                        metadata[model_type] = {}
                    if date not in metadata[model_type]:
                        metadata[model_type][date] = {}
                    if class_name not in metadata[model_type][date]:
                        metadata[model_type][date][class_name] = []

                    metadata[model_type][date][class_name].append(data)

    return metadata



if __name__ == "__main__":
    socketio.run(app, debug=False, allow_unsafe_werkzeug=True)

