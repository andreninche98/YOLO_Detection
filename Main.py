import time
from flask import Flask, render_template, request, jsonify, Response, send_from_directory, make_response
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
captured_frame_dict = manager.dict()
annotated_frame_dict = manager.dict()
capture_processes = []
capturing = False
stop_event = manager.Event()



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
detection_running = False


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
            p = multiprocessing.Process(target=capture_frames, args=(source, captured_frame_dict, stop_event))
            capture_processes.append(p)

        for p in capture_processes:
            p.start()
        raw_stream_running = True
        return jsonify(status='Raw stream started')
    else:
        return jsonify(status='Raw stream already running')


@app.route('/stop', methods=['POST'])
def stop_stream():
    global capture_processes, capturing, raw_stream_running, detection_running
    if raw_stream_running:
        capturing = False
        stop_event.set()
        detection_running = False
        for p in capture_processes:
            p.join()

        raw_stream_running = False
        return jsonify(status='Raw stream stopped')
    else:
        return jsonify(status='Raw stream not running')


@app.route('/detect', methods=['POST'])
def detect():
    global captured_frame_dict, annotated_stream_running, stop_event, detection_running
    if not annotated_stream_running:
        try:
            annotated_stream_running = True
            detection_running = True
            detection_processes = []
            for source in config['sources']:
                p = multiprocessing.Process(target=process_frame, args=(source, captured_frame_dict, batch_size))
                detection_processes.append(p)
                p.start()


            return jsonify({"status": "Detection started"}), 200

        except Exception as e:
            annotated_stream_running = False
            detection_running = False
            print(f"[ERROR] Error during detection: {e}")
            return jsonify({"status": "Error during detection", "error": str(e)}), 500
    else:
        return jsonify(status='Annotated stream already running')


@app.route('/status', methods=['GET'])
def status():
    global raw_stream_running, annotated_stream_running, detection_running
    return jsonify(raw_stream='Running' if raw_stream_running else 'Stopped',
                   annotated_stream='Running' if annotated_stream_running else 'Stopped',
                   detection='Detecting...' if detection_running else 'Detection stopped')


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

@app.route('/saved_frames/<path:filename>')
def serve_saved_frame(filename):
    return send_from_directory("/home/andrea/PycharmProjects/YOLO_Detection/frame_det", filename)


@app.route('/display_frame_with_bbox', methods=['GET'])
def display_frame_with_bbox():
    source_id = request.args.get('source_id')
    model_type = request.args.get('model_type')
    date = request.args.get('date')
    class_name = request.args.get('class_name')
    detection = request.args.get('detection')
    filename = request.args.get('filename')
    bbox = request.args.get('bbox')

    if not filename.endswith('.jpg'):
        return "Error: The requested file is not a .jpg file", 400

    img_path = f"/saved_frames/{source_id}/{model_type}/{date}/{class_name}/{detection}/{filename}"
    img = cv2.imread(img_path)

    if img is None:
        return "Error: The image file could not be read", 400


    xmin, ymin, xmax, ymax = map(int, bbox.split(','))
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    _, img_encoded = cv2.imencode('.jpg', img)
    response = make_response(img_encoded.tobytes())
    response.headers.set('Content-Type', 'image/jpg')

    return response


def capture_frames(source, captured_frame_dict, stop_event):
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
            if len(frames_list) >= 50:
                frames_list.pop(0)

            captured_frame_dict[source['id']] = frames_list

        print(f"[INFO] Finished capture from source: {source['type'], source['id']}")
        cap.release()

    except ValueError as ve:
        print(f"[ERROR] {ve}")
    except Exception as e:
        print(f"[ERROR] Error capturing frames: {e}")


def detect_with_yolo(frames, save_path, source_id):
    try:
        global annotated_frame_dict
        detections = Detection.detect_batch(frames, Detection.models)
        print(f"[INFO] Processed {len(frames)} frames for detection from source: {source_id}")
        for annotated_frame, metadata in detections:
            annotated_frame_dict[source_id] = (annotated_frame, metadata)
            if len(detections) >= batch_size:
                frame_handler.add_detection_job(frames, metadata, save_path, str(source_id))
        del captured_frame_dict[source_id][:len(frames)]
        remaining_frames = len(captured_frame_dict[source_id]) - len(frames)
        print(f"[INFO] Remaining frames for detection from source {source_id}: {remaining_frames}")
        return True
    except Exception as e:
        print(f"[ERROR] Error detecting with yolo: {e}")
        return False


def process_frame(source, captured_frame_dict, batch_size):
    source_id = source['id']
    if source_id not in captured_frame_dict or len(captured_frame_dict[source_id]) == 0:
        print(f"[WARNING] No frames captured from source: {source}")
    else:
        source_frames = captured_frame_dict[source_id]
        for i in range(0, len(source_frames), batch_size):
            batch = source_frames[i:i + batch_size]
            print(f"[INFO] Processing batch for {source['save_directory']}")
            try:
                detect_with_yolo(batch, source['save_directory'], source_id)
            except Exception as e:
                print(f"[ERROR] Error during detection batch processing: {e}")


def generate_raw_mjpeg_stream(frames_dict, source_id):
    global captured_frame_dict, stop_event

    try:
        while not stop_event.is_set():
            if raw_stream_running:
                    if source_id in captured_frame_dict and len(captured_frame_dict[source_id]) > 0:
                        time.sleep(0.1)
                        frame = captured_frame_dict[source_id][-1]  # Prendi l'ultimo frame catturato
                        _, jpeg = cv2.imencode('.jpg', frame)
                        frame = jpeg.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    except Exception as e:
        print(f"[ERROR] Error generating raw MJPEG stream: {e}")
        return


def generate_annotated_mjpeg_stream(frames_dict, source_id):
    global annotated_frame_dict, stop_event
    try:
        while not stop_event.is_set():
            if annotated_stream_running:
                if source_id in annotated_frame_dict.keys():
                    try:
                        time.sleep(0.1)
                        annotated_frame = annotated_frame_dict[source_id][0]
                        _, jpeg = cv2.imencode('.jpg', annotated_frame)
                        frame = jpeg.tobytes()
                        del annotated_frame_dict[source_id]
                    except Exception as e:
                        print(f"[ERROR] Exception occurred: {e}")
                        return
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    except Exception as e:
        print(f"[ERROR] Error generating annotated MJPEG stream: {e}")
        return


@app.route('/raw_stream/<int:source_id>')
def raw_stream(source_id):
    return Response(generate_raw_mjpeg_stream(captured_frame_dict, source_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/annotated_stream/<int:source_id>')
def annotated_stream(source_id):
    return Response(generate_annotated_mjpeg_stream(annotated_frame_dict, source_id), mimetype='multipart/x-mixed-replace; boundary=frame')



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

