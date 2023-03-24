## *Run the below script, camera will open and detect the Emotions. To exit just press 'q'*

from keras.models import load_model
from keras.utils import img_to_array
from mlxtend.image import extract_face_landmarks
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np
import time



def eye_aspect_ratio(eye):
    A = np.sqrt((float(eye[1][0])-float(eye[5][0]))**2 + (float(eye[1][1])-float(eye[5][1]))**2)
    B = np.sqrt((float(eye[2][0])-float(eye[4][0]))**2 + (float(eye[2][1])-float(eye[4][1]))**2)
    C = np.sqrt((float(eye[0][0])-float(eye[3][0]))**2 + (float(eye[0][1])-float(eye[3][1]))**2)

    ear = (A + B) / (2.0 * C)
    return ear
def mouth_aspect_ratio(mouth):
    
    A = np.sqrt((float(mouth[13][0])-float(mouth[19][0]))**2 + (float(mouth[13][1])-float(mouth[19][1]))**2)
    B = np.sqrt((float(mouth[14][0])-float(mouth[18][0]))**2 + (float(mouth[14][1])-float(mouth[18][1]))**2)
    C = np.sqrt((float(mouth[15][0])-float(mouth[17][0]))**2 + (float(mouth[15][1])-float(mouth[17][1]))**2)
    D = np.sqrt((float(mouth[12][0])-float(mouth[16][0]))**2 + (float(mouth[12][1])-float(mouth[16][1]))**2)
    
    mar = (A + B + C) / (3.0 * D)
    return mar
def nose_aspect_ratio(nose, landmarks):
    # Define the indexes of the eye landmarks
    LEFT_EYE_START_INDEX = 36
    LEFT_EYE_END_INDEX = 41
    RIGHT_EYE_START_INDEX = 42
    RIGHT_EYE_END_INDEX = 47
    
    A = np.sqrt((float(landmarks[36][0])-float(landmarks[41][0]))**2 + (float(landmarks[36][1])-float(landmarks[41][1]))**2)
    B = np.sqrt((float(landmarks[42][0])-float(landmarks[47][0]))**2 + (float(landmarks[42][1])-float(landmarks[47][1]))**2)
    interocular_distance = (A + B) / 2

    
    C = np.sqrt((float(nose[0][0])-float(nose[4][0]))**2 + (float(nose[0][1])-float(nose[4][1]))**2)
    nar = C / interocular_distance
    return nar

# extract features

def extract_features(landmarks):
    
    # distance between the corners of the eyes
    eye_dist = float(landmarks[45][0]) - float(landmarks[36][0])

    # Eyebrow position
    left_eyebrow_position = (float(landmarks[21][1]) + float(landmarks[22][1])) / 2
    right_eyebrow_position = (float(landmarks[1][1]) + float(landmarks[18][1])) / 2

    # Angle between the eyebrows
    eyebrow_angle = np.arctan2(float(landmarks[21][1]) - float(landmarks[17][1]), float(landmarks[22][0]) - float(landmarks[18][0])) * 180 / np.pi

    # Mouth height and width
    mouth_height = float(landmarks[66][1]) - float(landmarks[62][1])
    mouth_width = float(landmarks[54][0]) - float(landmarks[48][0])

    # Wrinkles around the eyes
    left_eye_crow_feet = float(landmarks[41][0]) - float(landmarks[36][0])
    right_eye_crow_feet = float(landmarks[45][0]) - float(landmarks[42][0])
    
    # Eye openness
    left_eye_height = float(landmarks[40][1]) - float(landmarks[38][1])
    right_eye_height = float(landmarks[47][1]) - float(landmarks[43][1])

    # Nose wrinkles
    nose_wrinkles = float(landmarks[31][1]) - float(landmarks[35][1])

    # Jaw tension
    jaw_tension = float(landmarks[57][1]) - float(landmarks[8][1])

    # Brow furrows
    brow_furrows = (float(landmarks[22][1]) - float(landmarks[21][1])) + (float(landmarks[27][1]) - float(landmarks[24][1]))

    # Chin thrust
    chin_thrust = float(landmarks[8][1]) - float(landmarks[51][1])

    # eye aspect ratio
    left_EAR = eye_aspect_ratio(landmarks[36:42])
    right_EAR = eye_aspect_ratio(landmarks[42:48])

    # mouth aspect ratio
    MAR = mouth_aspect_ratio(landmarks[48:68])

    # node aspect ratio
    NAR = nose_aspect_ratio(landmarks[31:36], landmarks)

    return [eye_dist, left_eyebrow_position, right_eyebrow_position, abs(round(eyebrow_angle, 3)), mouth_height, mouth_width, 
            left_eye_crow_feet, right_eye_crow_feet, left_eye_height, right_eye_height, abs(nose_wrinkles), abs(jaw_tension), 
            brow_furrows, chin_thrust, round(left_EAR,2), round(right_EAR, 3), round(MAR, 3), round(NAR, 3)]

def image(path):

    img = cv2.imread(path, 0)
    faces = face_classifier.detectMultiScale(img, 1.3, 10) # (scale factor, minNeighbors)
    for (x, y, w, h) in faces:
        img = img[y:y+h, x:x+w]
        img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
        
        landmarks = extract_face_landmarks(img)
        features = extract_features(landmarks)
        features = scaler.fit_transform([features])

        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        print(img.shape)
        prediction = classifier_features.predict([img, features])[0]
        label = emotion_labels[prediction.argmax()]
        print(label)

def cam():

    cap = cv2.VideoCapture(0) # default camera of the laptop

    while True:
        success, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 10) # (scale factor, minNeighbors)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (96, 96), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                try:
                    # a lag 
                    time.sleep(0.1)

                    landmarks = extract_face_landmarks(roi_gray)
                    features = extract_features(landmarks)
                    features = scaler.fit_transform([features])

                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    print(roi.shape)

                    prediction = classifier_features.predict([roi, features])[0]
                    label = emotion_labels[prediction.argmax()]
                    label_position = (x, y-10)
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)      

                except:
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    print(roi.shape)

                    prediction = classifier_normal.predict(roi)[0]
                    label = emotion_labels[prediction.argmax()]
                    label_position = (x, y-10)
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
            else:
                cv2.putText(frame, 'No Faces', label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

###########################################################################################################################

if __name__ ==  '__main__':

    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    scaler = StandardScaler()

    # loading the trained model
    classifier_features = load_model('EmotionDetectionFeat(87-81).h5')
    classifier_normal = load_model('EmotionDetectionModelAugBal.h5')
    emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']

    # uncomment for image input(input image path)
    #image(path)

    # for cam input
    cam()

    