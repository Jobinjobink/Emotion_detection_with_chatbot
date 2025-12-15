from flask import Flask, render_template, Response, request, jsonify
import cv2

from fer_model import detect_emotion_for_5_seconds
from chatbot import chatbot_response

app = Flask(__name__)

camera = cv2.VideoCapture(0)
CURRENT_EMOTION = "Neutral"

def gen_frames():
    global camera
    while True:
        success, frame = camera.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/start_scan")
def start_scan():
    global CURRENT_EMOTION
    CURRENT_EMOTION = detect_emotion_for_5_seconds()
    return jsonify({"emotion": CURRENT_EMOTION})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("message", "")
    reply = chatbot_response(msg, CURRENT_EMOTION)
    return jsonify({"reply": reply, "emotion": CURRENT_EMOTION})

if __name__ == "__main__":
    app.run(debug=True)
