from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import threading
import pandas as pd
import os
import datetime
from test import FaceAttendanceSystem

app = Flask(__name__)

system = FaceAttendanceSystem()
#source = 0
source = "http://192.168.29.167:4747/video"
cap = None
lock = threading.Lock()
camera_active = False

def generate_frames():
    global cap, camera_active
    with lock:
        if camera_active and (cap is None or not cap.isOpened()):
             cap = cv2.VideoCapture(source)

    while camera_active:
        with lock:
            if cap is None or not cap.isOpened():
                if camera_active:
                    cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    cv2.waitKey(100)
                    continue

            success, frame = cap.read()
            if not success:
                cap.release()
                continue
        
        try:
            frame = cv2.resize(frame, (640, 480))
            
            annotated_frame, detected_names = system.process_frame(frame)
            
            known_detected = any(name != "Unknown" for name in detected_names)
            
            if known_detected:
                camera_active = False
            
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if not ret: continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            if not camera_active:
                with lock:
                    if cap is not None:
                        cap.release()
                break

        except Exception as e:
            print(f"Error processing frame: {e}")
            continue
    
    with lock:
        if cap is not None and cap.isOpened():
            cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_scan', methods=['POST'])
def start_scan():
    global camera_active
    camera_active = True
    return jsonify({'success': True, 'message': 'Scanning started'})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_scan_status')
def check_scan_status():
    global camera_active
    return jsonify({'active': camera_active})

@app.route('/get_logs')
def get_logs():
    log_file = "attendance_log.csv"
    if os.path.exists(log_file):
        try:
            df = pd.read_csv(log_file)
            # Ensure columns exist
            expected_cols = ['Name', 'Date', 'EntryTime', 'ExitTime']
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = "" # Fill missing columns
            
            df = df.fillna('')
            return jsonify(df.to_dict(orient='records'))
        except Exception as e:
            print(f"Error reading logs: {e}") 
            return jsonify([])
    return jsonify([])

@app.route('/ask_bot', methods=['POST'])
def ask_bot():
    data = request.json
    query = data.get('query', '').lower()
    
    log_file = "attendance_log.csv"
    if not os.path.exists(log_file):
        return jsonify({'answer': "I have no attendance records yet."})
        
    try:
        df = pd.read_csv(log_file)
        df = df.fillna('')
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        today_df = df[df['Date'] == today]
        
        if "who is here" in query or "active" in query:
            active = today_df[today_df['ExitTime'] == 'Active']['Name'].unique()
            if len(active) > 0:
                return jsonify({'answer': f"Currently active: {', '.join(active)}"})
            else:
                return jsonify({'answer': "No one is currently active."})
                
        elif "how many" in query:
            count = len(today_df['Name'].unique())
            return jsonify({'answer': f"Total {count} unique people visited today."})
            
        elif "list" in query or "who visited" in query:
            names = today_df['Name'].unique()
            if len(names) > 0:
                return jsonify({'answer': f"Visitors today: {', '.join(names)}"})
            else:
                return jsonify({'answer': "No visitors recorded today."})
        
        else:
            return jsonify({'answer': "I can answer: 'Who is here?', 'How many people?', or 'List visitors'."})
            
    except Exception as e:
        return jsonify({'answer': f"Error processing query: {str(e)}"})

@app.route('/register_unknown', methods=['POST'])
def register_unknown():
    name = request.form.get('name')
    if not name: return jsonify({'success': False, 'message': 'Name is required'})
    success, message = system.register_last_unknown(name)
    return jsonify({'success': success, 'message': message})

@app.route('/delete_user', methods=['POST'])
def delete_user():
    name = request.form.get('name')
    if not name: return jsonify({'success': False, 'message': 'Name is required'})
    success, message = system.delete_user(name)
    return jsonify({'success': success, 'message': message})

@app.route('/export_logs')
def export_logs():
    log_file = "attendance_log.csv"
    if os.path.exists(log_file):
        try:
            df = pd.read_csv(log_file)
            df.to_excel("attendance_log.xlsx", index=False)
            return send_file("attendance_log.xlsx", as_attachment=True)
        except Exception as e: return str(e), 500
    return "No logs found", 404

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
