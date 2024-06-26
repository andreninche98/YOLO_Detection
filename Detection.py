import cv2
import torch
from datetime import datetime
import yaml
from torchvision import transforms
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_segments


with open("/home/andrea/PycharmProjects/YOLO_Detection/class_model_mapping.yaml", "r") as f:
    config = yaml.safe_load(f)

models = {}
class_names = {}
classes_to_detect = {}
for model_config in config["models"]:
    model_name = model_config["name"]
    model_file = model_config["file"]
    class_file = model_config["class_file"]
    try:
        models[model_name] = attempt_load(model_file)
        with open(class_file, "r") as f:
            data = yaml.safe_load(f)
            class_names[model_name] = data["names"]
        classes_to_detect[model_name] = {class_config["name"]: class_config["confidence_threshold"] for class_config in model_config["classes_to_detect"]}
    except Exception as e:
        print(f"Error loading model {model_file}: {e}")


# Trasformazioni per il preprocessing delle immagini
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Funzione per fare la detection su un frame
def detect(frame, model, model_name):
    original_frame = frame.copy()
    img_tens = cv2.resize(original_frame, (320, 320))
    img = transform(img_tens).unsqueeze(0)  # Prepara l'immagine per YOLOv5
    pred = model(img)[0]  # Esegui la prediction
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)[0] # Rimuovi le detection con bassa confidence
    metadata = []
    if pred is not None and len(pred) > 0:
        frame = img_tens.copy()
        img_shape = frame.shape[:2]
        pred[:, :4] = scale_segments(img_shape, pred[:, :4], img.shape[2:])  # Ridimensiona le coordinate
        for det in pred:
            det_np = det.cpu().numpy()
            xmin, ymin, xmax, ymax = det_np[:4].astype(int)
            conf = float(det_np[4])
            cls = det_np[5].astype(int)
            try:
                class_name = class_names[model_name][cls]
            except KeyError:
                print(
                    f"Warning: Class index {cls} not found in class file for model {model_name}. Skipping this detection.")
                continue
            class_confidence_threshold = classes_to_detect[model_name].get(class_name, 0.5)
            if conf < class_confidence_threshold and class_name in classes_to_detect[model_name]:
                continue
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Disegna il rettangolo intorno all'oggetto
            cv2.putText(frame, f'{class_name}: {conf:.2f} ({model_name})', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 2)  # Aggiungi il label dell'oggetto
            metadata.append({
                    "class_name": class_name,
                    "bbox": [xmin, ymin, xmax, ymax],
                    "confidence": float(f"{conf:.2f}"),
                    "model_type": model_name,
                    "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
                })
    return frame, metadata


def detect_batch(frames, model_names):
    try:
        batch_results = []
        for frame in frames:
            for model_name in model_names:
                model = models[model_name]
                annotated_frame, metadata = detect(frame, model, model_name)
                batch_results.append((annotated_frame, metadata))
        return batch_results
    except Exception as e:
        print(f"Error det batch: {e}")
        return []
