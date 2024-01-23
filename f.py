import multiprocessing
import queue
import time
import numpy as np
import cv2
import tensorflow.keras as tf
import pyttsx3
import math
import os
import mediapipe as mp

# Funzione per il rilevamento della mano
def hand_tracking(frame, classes, predictions, conf_threshold, speakQ):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = np.array([[int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])] for landmark in
                                  hand_landmarks.landmark])

            x, y, w, h = cv2.boundingRect(landmarks)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            for landmark in landmarks:
                cv2.circle(frame, tuple(landmark), 5, (0, 255, 0), -1)

            try:
                label = speakQ.get_nowait()  # Ottieni il messaggio dalla coda condivisa
                cv2.putText(frame, label, (x + 10, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            except multiprocessing.queues.Empty:
                pass

    return frame

# Funzione per ottenere il label del riconoscimento
def get_label(classes, current_prediction, conf_threshold):
    max_conf_index = np.argmax(current_prediction)
    max_confidence = current_prediction[max_conf_index]

    # Utilizziamo np.amax per ottenere il massimo valore nell'array di previsioni
    max_value = np.amax(current_prediction)

    if max_value > conf_threshold:
        return f"{classes[max_conf_index]}: {int(max_value * 100)}%"
    else:
        return "No Hand Detected"

# Funzione per la sintesi vocale e rilevamento della mano
def speak(speakQ):
    engine = pyttsx3.init()
    volume = engine.getProperty('volume')
    engine.setProperty('volume', volume)
    last_msg = ""
    while True:
        try:
            time.sleep(0.1)  # Aggiungi un breve ritardo qui
            msg = speakQ.get_nowait()
            if msg != last_msg and msg != "Background":
                last_msg = msg
                engine.say(msg)
                engine.runAndWait()
            if msg == "Background":
                last_msg = ""
        except multiprocessing.queues.Empty:
            pass

def main():
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    labels_path = f"{DIR_PATH}/model/labels.txt"
    labelsfile = open(labels_path, 'r')

    classes = []
    line = labelsfile.readline()
    while line:
        classes.append(line.split(' ', 1)[1].rstrip())
        line = labelsfile.readline()
    labelsfile.close()

    model_path = f"{DIR_PATH}/model/keras_model.h5"
    model = tf.models.load_model(model_path, compile=False)

    cap = cv2.VideoCapture(0)
    frameWidth = 1280
    frameHeight = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    cap.set(cv2.CAP_PROP_GAIN, 0)

    speakQ = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=speak, args=(speakQ,), daemon=True)
    p1.start()

    last_prediction = None  # Aggiungi questa variabile all'inizio del tuo codice

    while True:
        np.set_printoptions(suppress=True)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        check, frame = cap.read()
        margin = int(((frameWidth - frameHeight) / 2))
        square_frame = frame[0:frameHeight, margin:margin + frameHeight]
        resized_img = cv2.resize(square_frame, (224, 224))
        model_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        image_array = np.asarray(model_img)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        predictions = model.predict(data)

        # Solo se la prediction cambia, aggiorna il label dell'hand tracker
        if last_prediction is None or not np.array_equal(predictions, last_prediction):
            conf_threshold = 90
            confidence = []
            conf_label = ""
            threshold_class = ""

            bordered_frame = cv2.copyMakeBorder(
                square_frame,
                top=0,
                bottom=30 + 15 * math.ceil(len(classes) / 2),
                left=0,
                right=0,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )

            for i in range(0, len(classes)):
                confidence.append(int(predictions[0][i] * 100))
                if (i != 0 and not i % 2):
                    cv2.putText(
                        img=bordered_frame,
                        text=conf_label,
                        org=(int(0), int(frameHeight + 25 + 15 * math.ceil(i / 2))),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 255, 255)
                    )
                    conf_label = ""
                conf_label += classes[i] + ": " + str(confidence[i]) + "%; "
                if (i == (len(classes) - 1)):
                    cv2.putText(
                        img=bordered_frame,
                        text=conf_label,
                        org=(int(0), int(frameHeight + 25 + 15 * math.ceil((i + 1) / 2))),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 255, 255)
                    )
                    conf_label = ""
                if confidence[i] > conf_threshold:
                    speakQ.put(classes[i])
                    threshold_class = classes[i]

            cv2.putText(
                img=bordered_frame,
                text=threshold_class,
                org=(int(0), int(frameHeight + 20)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=(255, 255, 255)
            )

            # Esegui il rilevamento della mano e aggiorna il frame
            frame_with_hand = hand_tracking(bordered_frame, classes, predictions[0], conf_threshold, speakQ)

            cv2.imshow("Capturing", frame_with_hand)
            cv2.waitKey(10)

            last_prediction = predictions.copy()

    p1.terminate()

if __name__ == '__main__':
    main()
