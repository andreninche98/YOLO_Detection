import cv2
import torch
from datetime import datetime
import yaml
from torchvision import transforms
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_segments


def load_models():
    try:
        model_yolo = attempt_load("yolov5s.pt")  # Carica il modello YOLOv5
        model_tiny = attempt_load("yolov5n.pt")
        model_mid = attempt_load("yolov5m.pt")
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None
    return model_yolo, model_tiny, model_mid

models = load_models()

# Trasformazioni per il preprocessing delle immagini
transform = transforms.Compose([
    transforms.ToTensor(),
])

with open("/home/andrea/PycharmProjects/VideoDetection/.venv/lib/python3.10/coco.yaml", "r") as f:
    data = yaml.safe_load(f)
    class_names = data["names"]

class_model_mapping = {
    "person": "YOLO",
    "bed": "YOLO",
    "bottle": "TINY",
    "clock": "TINY",
}

classes_to_discard = ["car", "truck"]
# Funzione per fare la detection su un frame
def detect(frame, model, model_type):
    original_frame = frame.copy()
    img_tens = cv2.resize(original_frame, (416, 416))
    img = transform(img_tens).unsqueeze(0)  # Prepara l'immagine per YOLOv5
    pred = model(img)[0]  # Esegui la prediction
    pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.5)[0] # Rimuovi le detection con bassa confidence
    metadata = []
    if pred is not None and len(pred) > 0:
        frame = original_frame.copy()
        img_shape = frame.shape[:2]
        pred[:, :4] = scale_segments(img_shape, pred[:, :4], img.shape[2:])  # Ridimensiona le coordinate
        for det in pred:
            xmin, ymin, xmax, ymax, conf, cls = det.cpu().numpy().astype(int)
            class_name = class_names[cls]
            if class_name in classes_to_discard:
                continue
            if class_name in class_model_mapping:
                expected_model_type = class_model_mapping[class_name]
                if model_type != expected_model_type:
                    continue
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Disegna il rettangolo intorno all'oggetto
            cv2.putText(frame, f'{class_name} ({model_type})', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 2)  # Aggiungi il label dell'oggetto
            selected_model_type = class_model_mapping.get(class_name.lower(), "MID")
            metadata.append({
                    "class_name": class_name,
                    "bbox": [xmin, ymin, xmax, ymax],
                    "confidence": float(conf),
                    "model_type": selected_model_type,
                    "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
                })
    return metadata


def detect_batch(frames, models):
    try:
        batch_results = []
        for frame in frames:
            for model, model_type in zip(models, ["YOLO", "TINY", "MID"]):
                result_frame = detect(frame, model, model_type)
                batch_results.extend(result_frame)
        return batch_results
    except Exception as e:
        print(f"Error det batch: {e}")
        return []
