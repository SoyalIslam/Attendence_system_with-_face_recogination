import cv2
import os
import datetime
import torch
import numpy as np
import argparse
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

KNOWN_FACES_DIR = "known_faces"
LOG_FILE = "attendance_log.csv"
CONFIDENCE_THRESHOLD = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class FaceAttendanceSystem:
    def __init__(self, model_path=None):
        self.known_face_embeddings = []
        self.known_face_names = []
        self.last_unknown_crop = None 
        self.yolo_model = None
        self.resnet = None
        self.mtcnn = None
        
        self.active_sessions = {}
        self.EXIT_THRESHOLD = 5.0
        
        self.setup_models(model_path)
        self.load_known_faces()
        self.initialize_log_file()

    def initialize_log_file(self):
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w") as f:
                f.write("Name,Date,EntryTime,ExitTime\n")

    def setup_models(self, model_path=None):
        try:
            print(f"Initializing models on {DEVICE}...")
            if model_path is None:
                # Check for trained model first
                if os.path.exists("runs/detect/train/weights/best.pt"):
                    model_path = "runs/detect/train/weights/best.pt"
                elif os.path.exists("best.pt"):
                    model_path = "best.pt"
                else:
                    model_path = "yolov8n.pt"
            
            print(f"Using YOLO model: {model_path}")
            self.yolo_model = YOLO(model_path).to(DEVICE)
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
            self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=DEVICE)
        except Exception as e:
            print(f"Error initializing models: {e}")
            exit(1)

    def get_embedding(self, img_rgb):
        face = self.mtcnn(img_rgb)
        if face is not None:
            face = face.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                embedding = self.resnet(face)
            return embedding.cpu().numpy().flatten()
        return None

    def load_known_faces(self):
        print(f"Loading known faces from {KNOWN_FACES_DIR}...")
        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR)
        
        self.known_face_embeddings = []
        self.known_face_names = []

        for file in os.listdir(KNOWN_FACES_DIR):
            if file.endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(KNOWN_FACES_DIR, file)
                name = os.path.splitext(file)[0]
                try:
                    img = Image.open(path).convert('RGB')
                    emb = self.get_embedding(img)
                    if emb is not None:
                        self.known_face_embeddings.append(emb)
                        self.known_face_names.append(name)
                        print(f"Loaded: {name}")
                except Exception as e:
                    print(f"Could not load {file}: {e}")
        print(f"Total known faces: {len(self.known_face_names)}")

    def update_session(self, name, x_center):
        now = datetime.datetime.now()
        
        if name not in self.active_sessions:
#enter
            entry_time_str = now.strftime("%H:%M:%S")
            today_str = now.strftime("%Y-%m-%d")
            
            self.active_sessions[name] = {
                'entry_time': now,
                'last_seen': now,
                'last_x': x_center,
                'direction': "Entering"
            }
            
            with open(LOG_FILE, "a") as f:
                f.write(f"{name},{today_str},{entry_time_str},Active\n")
            print(f"Entry detected: {name}")
            
        else:
            session = self.active_sessions[name]
            
            dx = x_center - session['last_x']
            if abs(dx) < 2: direction = "Stationary"
            elif dx > 0: direction = "Moving Right ->"
            else: direction = "<- Moving Left"
            
            session['last_seen'] = now
            session['last_x'] = x_center
            session['direction'] = direction

    def check_exits(self):
        now = datetime.datetime.now()
        to_remove = []
        
        for name, session in self.active_sessions.items():
            if (now - session['last_seen']).total_seconds() > self.EXIT_THRESHOLD:
                exit_time_str = session['last_seen'].strftime("%H:%M:%S") # Use last seen as exit time
                self.log_exit(name, exit_time_str)
                to_remove.append(name)
                print(f"Exit detected: {name}")
        
        for name in to_remove:
            del self.active_sessions[name]

    def log_exit(self, name, exit_time):
        lines = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                lines = f.readlines()
        
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith(f"{name},") and "Active" in lines[i]:
                parts = lines[i].strip().split(',')
                if len(parts) >= 4:
                    parts[3] = exit_time
                    lines[i] = ",".join(parts) + "\n"
                    break
        
        with open(LOG_FILE, "w") as f:
            f.writelines(lines)

    def identify_face(self, img_rgb):
        if not self.known_face_embeddings:
            return "Unknown"

        emb = self.get_embedding(img_rgb)
        if emb is None:
            return "Unknown"

        best_match = "Unknown"
        min_dist = 0.6 

        for idx, known_emb in enumerate(self.known_face_embeddings):
            dist = 1 - np.dot(emb, known_emb) / (np.linalg.norm(emb) * np.linalg.norm(known_emb))
            if dist < min_dist:
                min_dist = dist
                best_match = self.known_face_names[idx]
                
        return best_match

    def register_last_unknown(self, new_name):
        if self.last_unknown_crop is None:
            return False, "No unknown face detected recently."
        
        if not new_name:
            return False, "Name cannot be empty."

        filename = f"{new_name}.jpg"
        filepath = os.path.join(KNOWN_FACES_DIR, filename)
        
        try:
            cv2.imwrite(filepath, self.last_unknown_crop)
            print(f"Saved new face: {filepath}")
            self.load_known_faces()
            self.update_session(new_name, 0) 
            return True, f"Successfully registered {new_name}"
        except Exception as e:
            return False, str(e)

    def delete_user(self, name):
        if not name:
            return False, "Name cannot be empty."

        found = False
        for file in os.listdir(KNOWN_FACES_DIR):
            if os.path.splitext(file)[0] == name:
                filepath = os.path.join(KNOWN_FACES_DIR, file)
                try:
                    os.remove(filepath)
                    found = True
                    break
                except Exception as e:
                    return False, f"Error deleting file: {str(e)}"
        
        if found:
            self.load_known_faces()
            if os.path.exists(LOG_FILE):
                try:
                    with open(LOG_FILE, "r") as f:
                        lines = f.readlines()
                    new_lines = [line for line in lines if not line.startswith(f"{name},")]
                    with open(LOG_FILE, "w") as f:
                        if not lines or not lines[0].startswith("Name"):
                             f.write("Name,Date,EntryTime,ExitTime\n")
                        f.writelines(new_lines)
                except Exception as e:
                    print(f"Error updating CSV: {e}")

            return True, f"User '{name}' deleted successfully."
        else:
            return False, f"User '{name}' not found."

    def process_frame(self, frame):
        results = self.yolo_model(frame, verbose=False)
        detected_names = []
        draw_queue = []

        for result in results:
            class_names = result.names
            
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = class_names[cls]

                if conf < CONFIDENCE_THRESHOLD:
                    continue

                if label == 'person':
                    person_crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if person_crop.size == 0:
                        continue

                    img_rgb = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                    name = self.identify_face(img_rgb)
                    detected_names.append(name)

                    if name == "Unknown":
                        self.last_unknown_crop = person_crop.copy()
                        color = (0, 0, 255)
                        bg_color = (0, 0, 150)
                        direction_text = ""
                    else:
                        color = (0, 255, 0)
                        bg_color = (0, 150, 0)
                        

                        x_center = (x1 + x2) // 2
                        self.update_session(name, x_center)
                        direction_text = self.active_sessions.get(name, {}).get('direction', '')
                    
                    draw_queue.append({
                        'coords': (x1, y1, x2, y2),
                        'name': name,
                        'color': color,
                        'bg_color': bg_color,
                        'direction': direction_text,
                        'is_person': True
                    })
                else:
                    draw_queue.append({
                        'coords': (x1, y1, x2, y2),
                        'name': label,
                        'color': (255, 165, 0),
                        'bg_color': (150, 100, 0),
                        'direction': f"{int(conf*100)}%",
                        'is_person': False
                    })

        self.check_exits()

        overlay = frame.copy()
        for item in draw_queue:
            x1, y1, x2, y2 = item['coords']
            bg_color = item['bg_color']
            name = item['name']
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
            
            (w, h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(overlay, (x1, y1 - 25), (x1 + w, y1), bg_color, -1)

        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        for item in draw_queue:
            x1, y1, x2, y2 = item['coords']
            color = item['color']
            name = item['name']
            direction = item['direction']

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            if direction:
                cv2.putText(frame, direction, (x1, y2 + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame, detected_names

def train_model(data_yaml="data.yaml", epochs=100):
    print(f"Starting training with {data_yaml} for {epochs} epochs...")
    model = YOLO("yolov8n.pt")
    results = model.train(data=data_yaml, epochs=epochs, imgsz=640)
    print("Training complete. Results saved in runs/detect/train/")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLO model")
    args = parser.parse_args()

    if args.train:
        if not os.path.exists(args.data):
            print(f"Error: {args.data} not found. Please create it first.")
            with open("data.yaml", "w") as f:
                f.write("train: datasets/train/images\nval: datasets/val/images\n\nnc: 4\nnames: ['person', 'helmet', 'cap', 'sunglasses']\n")
            print("A template 'data.yaml' has been created. Please prepare your dataset in 'datasets/' folder.")
            return
        train_model(args.data, args.epochs)
    else:
        system = FaceAttendanceSystem(model_path=args.model)
        cap = cv2.VideoCapture("https://192.168.29.167:4747/video")
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Starting Attendance System... Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame, _ = system.process_frame(frame)
            cv2.imshow("Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
