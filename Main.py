import time

from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_socketio import SocketIO, emit
import cv2
import multiprocessing
import json
import base64
import yaml
from pytube import YouTube
import os
import frame_handler
import Detection



app = Flask(__name__, template_folder='/home/andrea/PycharmProjects/YOLO_Detection/templates')
socketio = SocketIO(app)
manager = multiprocessing.Manager()
return_dict = manager.dict()
capture_processes = []
capturing = False
stop_event = manager.Event()
results = []


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

raw_stream_running = False
annotated_stream_running = False


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results_page():
    return render_template('result_page.html')

@app.route('/start', methods=['POST'])
def start_stream():
    global capture_processes, capturing, raw_stream_running
    if not raw_stream_running:
        capturing = True
        stop_event.clear()
        for p in worker_processes:
            p.start()

        for source in config['sources']:
            p = multiprocessing.Process(target=capture_frames, args=(source, return_dict, stop_event))
            capture_processes.append(p)

        for p in capture_processes:
            p.start()
        raw_stream_running = True
        return jsonify(status='Raw stream started')
    else:
        return jsonify(status='Raw stream already running')


@app.route('/stop', methods=['POST'])
def stop_stream():
    global capture_processes, capturing, raw_stream_running
    if raw_stream_running:
        capturing = False
        stop_event.set()
        for p in capture_processes:
            p.join()

        capture_processes.clear()
        raw_stream_running = False
        return jsonify(status='Raw stream stopped')
    else:
        return jsonify(status='Raw stream not running')


@app.route('/detect', methods=['POST'])
def detect():
    global return_dict, annotated_stream_running, stop_event
    if not annotated_stream_running:
        try:
            annotated_stream_running = True
            detection_processes = []
            for source in config['sources']:
                p = multiprocessing.Process(target=process_frame, args=(source, return_dict, batch_size))
                detection_processes.append(p)
                p.start()

            return jsonify({"status": "Detection started"}), 200

        except Exception as e:
            annotated_stream_running = False
            print(f"[ERROR] Error during detection: {e}")
            return jsonify({"status": "Error during detection", "error": str(e)}), 500
    else:
        return jsonify(status='Annotated stream already running')


@app.route('/status', methods=['GET'])
def status():
    global raw_stream_running, annotated_stream_running
    return jsonify(raw_stream='Running' if raw_stream_running else 'Stopped',
                   annotated_stream='Running' if annotated_stream_running else 'Stopped')


@app.route('/list_saved_frames')
def list_saved_frames():
    root_dir = "/home/andrea/PycharmProjects/YOLO_Detection/frame_det"
    frames = {}
    for source_id in os.listdir(root_dir):
        source_path = os.path.join(root_dir, source_id)
        if os.path.isdir(source_path):
            frames[source_id] = {}
            for model_type in os.listdir(source_path):
                model_path = os.path.join(source_path, model_type)
                if os.path.isdir(model_path):
                    frames[source_id][model_type] = {}
                    for date in os.listdir(model_path):
                        date_path = os.path.join(model_path, date)
                        if os.path.isdir(date_path):
                            frames[source_id][model_type][date] = {}
                            for class_name in os.listdir(date_path):
                                class_path = os.path.join(date_path, class_name)
                                if os.path.isdir(class_path):
                                    frames[source_id][model_type][date][class_name] = {}
                                    for detection in os.listdir(class_path):
                                        detection_path = os.path.join(class_path, detection)
                                        if os.path.isdir(detection_path):
                                            frames[source_id][model_type][date][class_name][detection] = os.listdir(detection_path)
    return jsonify(frames)

@app.route('/saved_frames/<source_id>/<model_type>/<date>/<class_name>/<filename>')
def serve_saved_frame(source_id, model_type, date, class_name, filename):
    return send_from_directory(os.path.join("/home/andrea/PycharmProjects/YOLO_Detection/frame_det", source_id, model_type, date, class_name), filename)



def capture_frames(source, return_dict, stop_event):
    try:
        frames_list = []
        print(f"[INFO] Starting capture from source: {source['type'], source['id']}")

        if source['type'] == 'webcam':
            cap = cv2.VideoCapture(source['id'])
        elif source['type'] == 'video':
            video_files = [f for f in os.listdir(source['save_directory']) if f.endswith('.mp4')]
            if video_files:
                video_path = os.path.join(source['save_directory'], video_files[0])
                cap = cv2.VideoCapture(video_path)
            else:
                video_path = download_youtube_video(source['id'], source['save_directory'])
                if not video_path:
                    raise ValueError(f"Failed to download video from URL: {source['id']}")
                cap = cv2.VideoCapture(video_path)
        else:
            raise ValueError(f"Unsupported source type: {source['type']}")

        if not cap.isOpened():
            raise ValueError(f"Failed to open source: {source['type']}")

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frames_list.append(frame)
            print(f"[INFO] Frame captured from source: {source['type'], source['id']}, {len(frames_list)} captured")

            return_dict[source['id']] = frames_list

        print(f"[INFO] Finished capture from source: {source['type'], source['id']}")
        cap.release()

    except ValueError as ve:
        print(f"[ERROR] {ve}")
    except Exception as e:
        print(f"[ERROR] Error capturing frames: {e}")


def detect_with_yolo(frames, save_path, source_id):
    try:
        global return_dict
        detections = Detection.detect_batch(frames, Detection.models)
        for annotated_frame, metadata in detections:
            return_dict[source_id] = (annotated_frame, metadata)
            frame_handler.add_detection_job(frames, metadata, save_path, str(source_id))
        return True
    except Exception as e:
        print(f"[ERROR] Error detecting with yolo: {e}")
        return False


def process_frame(source, return_dict, batch_size):
    source_id = source['id']
    if source_id not in return_dict or len(return_dict[source_id]) == 0:
        print(f"[WARNING] No frames captured from source: {source}")
    else:
        source_frames = return_dict[source_id]
        for i in range(0, len(source_frames), batch_size):
            batch = source_frames[i:i + batch_size]
            print(f"[INFO] Processing batch for {source['save_directory']}")
            try:
                detect_with_yolo(batch, source['save_directory'], source_id)
            except Exception as e:
                print(f"[ERROR] Error during detection batch processing: {e}")


def generate_raw_mjpeg_stream():
    global return_dict, stop_event

    try:
        while not stop_event.is_set():
            if raw_stream_running:
                for source in config['sources']:
                    source_id = source['id']
                    if source_id in return_dict and len(return_dict[source_id]) > 0:
                        time.sleep(0.1)
                        frame = return_dict[source_id][-1]  # Prendi l'ultimo frame catturato
                        _, jpeg = cv2.imencode('.jpg', frame)
                        frame = jpeg.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    except Exception as e:
        print(f"[ERROR] Error generating raw MJPEG stream: {e}")
        return


def generate_annotated_mjpeg_stream():
    global return_dict, stop_event
    try:
        while not stop_event.is_set():
            if annotated_stream_running:
                for source_id in return_dict.keys():
                    time.sleep(0.1)
                    try:
                        annotated_frame = return_dict[source_id][0]
                        _, jpeg = cv2.imencode('.jpg', annotated_frame)
                        frame = jpeg.tobytes()
                    except Exception as e:
                        print(f"[ERROR] Exception occurred: {e}")
                        return
                    print(f"[DEBUG] Sending annotated frame: Shape {len(frame)}")
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    except Exception as e:
        print(f"[ERROR] Error generating annotated MJPEG stream: {e}")
        return

@app.route('/raw_stream')
def raw_stream():
    return Response(generate_raw_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/annotated_stream')
def annotated_stream():
    return Response(generate_annotated_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


def load_saved_frames_metadata():
    try:
        metadata = {}
        save_path = "/home/andrea/PycharmProjects/YOLO_Detection/frame_det"

        for source_id in os.listdir(save_path):
            source_path = os.path.join(save_path, source_id)
            if not os.path.isdir(source_path):
                continue

            metadata[source_id] = {}

            for model_type in os.listdir(source_path):
                model_path = os.path.join(source_path, model_type)
                if not os.path.isdir(model_path):
                    continue

                metadata[source_id][model_type] = {}

                for date in os.listdir(model_path):
                    date_path = os.path.join(model_path, date)
                    if not os.path.isdir(date_path):
                        continue

                    metadata[source_id][model_type][date] = {}

                    for class_name in os.listdir(date_path):
                        class_path = os.path.join(date_path, class_name)
                        if not os.path.isdir(class_path):
                            continue

                        metadata[source_id][model_type][date][class_name] = []

                        for detection_folder in os.listdir(class_path):
                            detection_path = os.path.join(class_path, detection_folder)
                            if not os.path.isdir(detection_path):
                                continue

                            # Assume che ci sia un solo frame e un solo metadata per ogni detection
                            frame_files = [f for f in os.listdir(detection_path) if
                                           f.startswith('frame_') and f.endswith('.jpg')]
                            metadata_files = [f for f in os.listdir(detection_path) if
                                              f.startswith('metadata_') and f.endswith('.json')]

                            if len(frame_files) != 1 or len(metadata_files) != 1:
                                continue

                            frame_filename = os.path.join(detection_path, frame_files[0])
                            metadata_filename = os.path.join(detection_path, metadata_files[0])

                            try:
                                with open(metadata_filename, 'r') as f:
                                    data = json.load(f)
                                    data['frame_filename'] = frame_filename  # Aggiungi il percorso dell'immagine del frame
                                    metadata[source_id][model_type][date][class_name].append(data)
                            except Exception as e:
                                print(f"Error loading frame or metadata for {detection_folder}: {e}")

                return metadata
    except Exception as e:
        print(f"Error in load_saved_frames_metadata: {e}")
        raise

def download_youtube_video(url, save_path):
    try:
        print(f"[INFO] Downloading video from URL: {url}")
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
        if not stream:
            raise ValueError("No suitable stream found")
        stream.download(output_path=save_path, filename='downloaded_video.mp4')
        video_path = os.path.join(save_path, 'downloaded_video.mp4')
        print(f"[INFO] Video downloaded to: {video_path}")
        return video_path
    except Exception as e:
        print(f"[ERROR] Error downloading video: {e}")
        return None


if __name__ == "__main__":
    socketio.run(app, debug=False, allow_unsafe_werkzeug=True)

