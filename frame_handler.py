import os
from datetime import datetime
import cv2
from rq import Queue, Worker, Connection
from redis import Redis
import json


# Inizializza la connessione Redis
redis_conn = Redis()

# Creazione code RQ per modelli
queue_yolo = Queue("queue_yolo",connection=redis_conn)
queue_tiny = Queue("queue_tiny", connection=redis_conn)
queue_mid = Queue("queue_mid", connection=redis_conn)


# Funzione per salvare il frame su disco con un timestamp nel nome del file
def save_detection_metadata(metadata, frame, save_path, source_id):
    source_save_path = os.path.join(save_path, source_id)
    os.makedirs(source_save_path, exist_ok=True)
    model_save_path = os.path.join(source_save_path, metadata['model_type'])
    os.makedirs(model_save_path, exist_ok=True)
    current_date = datetime.now().strftime("%Y-%m-%d")
    date_save_path = os.path.join(model_save_path, current_date)
    os.makedirs(date_save_path, exist_ok=True)
    class_save_path = os.path.join(date_save_path, metadata['class_name'])
    os.makedirs(class_save_path, exist_ok=True)

    detection_folder = os.path.join(class_save_path, f'detection_{metadata["timestamp"]}')
    os.makedirs(detection_folder, exist_ok=True)

    frame_filename = os.path.join(detection_folder, f'frame_{metadata["timestamp"]}.jpg')
    cv2.imwrite(frame_filename, frame)

    detection_dict = {
            "class_name": str(metadata["class_name"]),
            "bbox": [int(coord) for coord in metadata["bbox"]],
            "confidence": float(metadata["confidence"]),
            "model_type": str(metadata["model_type"]),
            "timestamp": str(metadata["timestamp"]),
            "frame_filename": frame_filename
    }
    metadata_filename = os.path.join(detection_folder, f'metadata_{metadata["timestamp"]}.json')
    with open(metadata_filename, 'w') as f:
        json.dump(detection_dict, f, indent=4)
    print(f"[INFO] Saved detection to {metadata_filename}")
    print(f"[INFO] Saved detection to {frame_filename}")



def save_detection_metadata_batch(batch):
    for metadata, frame, save_path , source_id in batch:
        save_detection_metadata(metadata, frame, save_path, str(source_id))


# Funzione per aggiungere il lavoro alla coda RQ
def add_detection_job(frames, metadata_batch, save_path, source_id):
    source_id = str(source_id)
    yolo_batch = []
    tiny_batch = []
    mid_batch = []

    for frame, metadata in zip(frames, metadata_batch):
        model_type = metadata["model_type"]
        if model_type == "YOLO":
            yolo_batch.append((metadata, frame, save_path, source_id))
        elif model_type == "TINY":
            tiny_batch.append((metadata, frame, save_path, source_id))
        elif model_type == "MID":
            mid_batch.append((metadata, frame, save_path, source_id))

    # Inserisce le batch nelle rispettive code
    if yolo_batch:
        queue_yolo.enqueue(save_detection_metadata_batch, yolo_batch)
    if tiny_batch:
        queue_tiny.enqueue(save_detection_metadata_batch, tiny_batch)
    if mid_batch:
        queue_mid.enqueue(save_detection_metadata_batch, mid_batch)



class CustomWorker(Worker):
    def execute_job(self, job, queue):
        queues = [queue_yolo, queue_tiny, queue_mid]
        while len(queue) > 1:
            job_tuple = queue.dequeue_any(queues, timeout=1)
            if job_tuple:
                job_to_discard, _ = job_tuple
                job_to_discard.delete()
        super().execute_job(job, queue)

# Avvia i worker RQ
def start_worker(queue_name):
    with Connection(redis_conn):
        worker = CustomWorker([Queue(queue_name, connection=redis_conn)])
        worker.work()