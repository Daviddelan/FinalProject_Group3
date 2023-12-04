from flask import Flask, render_template, Response, request, redirect, url_for

import cv2
import mediapipe as mp
import numpy as np
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
import math
from keras.preprocessing import image
import tensorflow as tf
from keras.applications.vgg16 import VGG16

model = VGG16(weights='imagenet', include_top=False)

# from tensorflow.keras.applications.vgg16 import VGG16

base_model = VGG16(weights='imagenet', include_top=False)


app = Flask(__name__)

selected_exercise = 'push_ups'
counter = 0
stage = None 

# Load your model
model = load_model('/Users/daviddela/Intro to AI/AI Final Project/model.h5')  # Replace with your model's path


def stop():
    global camera_on
    camera_on.clear()  # Turn the camera off
    return render_template('index.html')  # Redirect back to the main page

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def is_down_stage(elbow_y, shoulder_y):
    """Check if the user is in the down stage of a push-up."""
    # Adjust the threshold as needed
    return elbow_y >= shoulder_y

def arms_raised(wrist_y, shoulder_y):
    return wrist_y < shoulder_y

# Function to check if legs are spread wider than hips
def legs_spread(ankle_y, hip_y):
    return ankle_y < hip_y

def full_body_visible(landmarks):
    mp_pose = mp.solutions.pose
    required_landmarks = [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE,
                          mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]
    return all(landmarks[landmark.value].visibility > 0.5 for landmark in required_landmarks)

# Function to determine if a jump is occurring
def is_jumping(ankles, prev_ankles, wrists, prev_wrists, jump_threshold=0.01, hand_threshold=0.01):
    # Check significant vertical movement of ankles and wrists
    ankles_moved_up = (prev_ankles - ankles) > jump_threshold
    wrists_moved_down = (wrists - prev_wrists) > hand_threshold
    return ankles_moved_up and wrists_moved_down

def calculate_speed(current_pos, prev_pos, time_elapsed):
    distance = np.sqrt((current_pos.x - prev_pos.x) ** 2 + (current_pos.y - prev_pos.y) ** 2)
    speed = distance / time_elapsed if time_elapsed > 0 else 0
    return speed


def gen_frames():
    global selected_exercise
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    counter = 0
    stage = None

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                try:
                    landmarks = results.pose_landmarks.landmark
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
                    right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
                    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                    left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                    right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                    left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                    right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
                    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
                    ankles = (left_ankle + right_ankle) / 2
                    wrists = (left_wrist + right_wrist) / 2


                    angle = calculate_angle(shoulder, elbow, wrist)

                    if selected_exercise == 'curl_ups':
                        # Curl-ups specific logic
                        if angle > 160:
                            stage = "down"
                        elif angle < 30 and stage == 'down':
                            stage = "up"
                            counter += 1

                    elif selected_exercise == 'push_ups':
                        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]

                        if is_down_stage(elbow.y, shoulder.y) and stage == 'up':
                            stage = 'down'
                        elif not is_down_stage(elbow.y, shoulder.y) and stage == 'down':
                            stage = 'up'
                            counter += 1  


                    elif selected_exercise == 'jumping_jack':
                        if arms_raised(left_wrist_y, left_shoulder_y) and arms_raised(right_wrist_y, right_shoulder_y) and legs_spread(left_ankle_y, left_hip_y) and legs_spread(right_ankle_y, right_hip_y):
                            if stage == 'closed':
                                stage = 'open'
                        else:
                            if stage == 'open':
                                stage = 'closed'
                                counter += 1  # Count a jumping jack  

                    elif selected_exercise == 'sit_ups':
                        # Calculate shoulder-hip angle
                        angle = calculate_angle(left_shoulder, left_hip, right_hip)
                        # Sit-up detection logic
                        if angle < 45 and stage == 'down':  # Adjust the angle threshold as needed
                            stage = 'up'
                        elif angle > 70 and stage == 'up':  # Adjust the angle threshold as needed
                            stage = 'down'
                            counter += 1  # Count a sit-up
        
                    elif selected_exercise == 'skipping': 
                        if is_jumping(ankles, prev_ankles, wrists, prev_wrists):
                            counter += 1  # Count a jump
                        prev_ankles = ankles
                        prev_wrists = wrists     

                except:
                    pass

                cv2.putText(frame, f'Reps: {counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if selected_exercise != 'skipping_rope':
                    cv2.putText(frame, f'Stage: {stage}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def index():
    global selected_exercise, counter, stage
    if request.method == 'POST':
        selected_exercise = request.form.get('exercise')
        print("Selected Exercise:", selected_exercise)  # Debugging print statement
        counter = 0
        stage = None
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def extract_frames(video_path, frames_dir):
    cap = cv2.VideoCapture(video_path)
    frameRate = cap.get(5)  # frame rate
    count = 0
    while cap.isOpened():
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % math.floor(frameRate) == 0:
            filename = f"frame{count}.jpg"
            count += 1
            cv2.imwrite(os.path.join(frames_dir, filename), frame)
    cap.release()


# def process_video_and_predict(video_path):

#     frames_dir = '/Users/daviddela/Downloads/trials/frames'
#     os.makedirs(frames_dir, exist_ok=True)

#     # Extract frames
#     extract_frames(video_path, frames_dir)

#     # Preprocess frames and predict
#     all_predictions = []
#     for frame_name in os.listdir(frames_dir):
#         frame_path = os.path.join(frames_dir, frame_name)
#         preprocessed_features = preprocess_and_extract_features(frame_path, vgg_base)
#         prediction = model.predict(preprocessed_features)
#         all_predictions.append(prediction)
#         average_prediction = np.mean(np.array(all_predictions), axis=0)

#         actions_list = [
#     'barbell_biceps_curl', 'bench_press', 'chest_fly_machine', 'deadlift', 'decline_bench_press', 'hammer_curl', 
#     'hip_thrust', 'incline_bench_press', 'lat_pulldown', 'lateral_raise', 'leg_extension', 'leg_raises', 'plank', 
#     'pull_up', 'push_up', 'romanian_deadlift', 'russian_twist', 'shoulder_press', 'squat', 't_bar_row', 'tricep_dips', 
#     'tricep_pushdown'
#         ]


#         predicted_class = np.argmax(average_prediction, axis=1)[0]
#         predicted_exercise = actions_list[predicted_class]
#         print("\n\n\n\n\n\n\n\n\n\n\n\n"+predicted_exercise)

#     return predicted_exercise

def process_video_and_predict(video_path):
    frames_dir = '/Users/daviddela/Downloads/trials/frames'
    os.makedirs(frames_dir, exist_ok=True)

    # Extract frames
    extract_frames(video_path, frames_dir)

    # Preprocess frames and predict
    all_predictions = []
    for frame_name in os.listdir(frames_dir):
        try:
            frame_path = os.path.join(frames_dir, frame_name)
            preprocessed_features = preprocess_and_extract_features(frame_path, base_model)
            prediction = model.predict(preprocessed_features)
            all_predictions.append(prediction)
        except Exception as e:
            print(f"Error processing frame {frame_name}: {e}")

    # Aggregate predictions
    if all_predictions:
        average_prediction = np.mean(np.array(all_predictions), axis=0)
        actions_list = ['barbell_biceps_curl', 'bench_press', 'chest_fly_machine', 'deadlift', 'decline_bench_press', 'hammer_curl', 'hip_thrust', 'incline_bench_press', 'lat_pulldown', 'lateral_raise', 'leg_extension', 'leg_raises', 'plank', 'pull_up', 'push_up', 'romanian_deadlift', 'russian_twist', 'shoulder_press', 'squat', 't_bar_row', 'tricep_dips', 'tricep_pushdown']
     # Add all your actions here
        predicted_class = np.argmax(average_prediction, axis=1)[0]
        predicted_exercise = actions_list[predicted_class]
        print(f"Predicted Exercise: {predicted_exercise}")
        return predicted_exercise
    else:
        print("No predictions made")
        return None


    # Process all_predictions as needed


def preprocess_and_extract_features(frame_path, base_model):
    # Load and preprocess the image
    img = image.load_img(frame_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Extract features using the base model
    features = base_model.predict(img_array)

    # Flatten the features for feeding into the dense layers
    flattened_features = features.reshape(1, -1)

    return flattened_features



@app.route('/storage', methods=['GET', 'POST'])
# def storage():
#     if request.method == 'POST':
#         video = request.files['video']
#         if video.filename == '':
#             return redirect(request.url)
#         if video:
#             filename = secure_filename(video.filename)
#             video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             video.save(video_path)

#             # Call the function to process video
#             process_video_and_predict(video_path)

#             return render_template('results.html')  # Assuming you have a result template

#     return render_template('storage.html')
@app.route('/storage', methods=['GET', 'POST'])
def storage():
    if request.method == 'POST':
        video = request.files['video']
        if video.filename == '':
            return redirect(request.url)
        if video:
            filename = secure_filename(video.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video.save(video_path)

            predicted_exercise = process_video_and_predict(video_path)
            return render_template('results.html', predicted_exercise=predicted_exercise)

    return render_template('storage.html')




app.config['UPLOAD_FOLDER'] = '/Users/daviddela/Downloads/trials/VideoUploads'  # Set your desired upload folder

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:  # Check if the file is valid (you might want to check file extensions)
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Process the video file as needed
        return render_template('results.html')  # Redirect after processing




if __name__ == '__main__':
    app.run(debug=True)
