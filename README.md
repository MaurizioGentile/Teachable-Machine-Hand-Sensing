# Teachable-Machine-Hand-Sensing
This repository is dedicated to the development of a hand tracking system using Teachable Machine to recognize and interpret sign language letters for the deaf-mute community. The primary goal is to create an interactive application that translates hand gestures into corresponding letters, facilitating communication for individuals with hearing and vocal impairments.

Key Features:

Hand Tracking: Implementation of a real-time hand tracking system using technologies like OpenCV to capture and monitor hand movements.

Integration with Teachable Machine: Utilization of Teachable Machine, a machine learning framework developed by Google, to train the model to recognize specific gestures associated with each letter of the alphabet.

Letter Recognition: Implementation of algorithms to interpret hand signals and map them to the corresponding letters in the sign language alphabet.

Interactive User Interface: Creation of a user-friendly interface allowing users to interact with the system intuitively, displaying recognized letters and facilitating communication.

Detailed Documentation: Provision of comprehensive documentation explaining the usage of the repository, system configuration, and instructions for additional Teachable Machine model training.

Open Source License: The repository is made available under an open-source license to promote collaboration and knowledge-sharing in the field of accessibility and human-computer interaction.

Demo Purpose:
This project serves as a demonstration of the capabilities of the Teachable Machine Hand Sensing system. It showcases how innovative technologies can be applied to improve accessibility, offering a tool for individuals with hearing and vocal disabilities to communicate effectively through sign language gestures.


### Teachable Machine Hand Sensing Demo

---

## Overview

This repository contains a Python script for real-time hand tracking, sign language recognition, and text-to-speech synthesis. The script utilizes OpenCV, TensorFlow, Mediapipe, and pyttsx3.

### Installation

```bash
pip install opencv-python tensorflow pyttsx3 mediapipe
```

### Running the Script

```bash
python hand_tracking_demo.py
```

## Code Explanation

### Hand Tracking Function

The `hand_tracking` function utilizes the MediaPipe library to detect and track hand landmarks in real-time. It draws bounding boxes around the detected hands and displays the recognized letters.

```python
# Hand detection function
def hand_tracking(frame, classes, predictions, conf_threshold, speakQ):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # ... (MediaPipe hand tracking code)
    
    return frame
```

### Text-to-Speech Function

The `speak` function uses pyttsx3 for text-to-speech synthesis. It continuously checks the shared queue for messages and speaks them.

```python
# Function for speech synthesis and hand detection
def speak(speakQ):
    engine = pyttsx3.init()
    
    # ... (Code for adjusting volume and continuous speech synthesis)
    
    while True:
        try:
            time.sleep(0.1)  # (new) Add a short delay here
            msg = speakQ.get_nowait()
            if msg != last_msg and msg != "Background":
                # ... (Code for speaking the message)
            if msg == "Background":
                last_msg = ""
        except multiprocessing.queues.Empty:
            pass
```

### Main Function

The `main` function initializes the required paths, loads the Teachable Machine model, sets up video capture, and continuously processes frames for hand tracking and recognition.

```python
# Main function
def main():
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    
    # ... (Code for loading model, setting up video capture, and creating processes)
    
    while True:
        # ... (Code for capturing frames, resizing, and making predictions)
        
        # ... (Code for dynamic confidence thresholding and displaying information)
        
        # ... (Code for hand detection and updating the frame)
        
        # ... (Code for showing the frame)
        
        last_prediction = predictions.copy()
    
    p1.terminate()

if __name__ == '__main__':
    main()
```

### Configuration

- **Model and Labels:** Set the correct paths for the Teachable Machine model (`keras_model.h5`) and labels (`labels.txt`) in the `model` directory.

- **Frame Dimensions:** Adjust `frameWidth` and `frameHeight` based on your camera feed requirements.

### Usage

1. Run the script.
2. View the live video feed with hand tracking and letter recognition.
3. Audible feedback is provided through text-to-speech.

Feel free to explore and customize the script according to your specific needs.
