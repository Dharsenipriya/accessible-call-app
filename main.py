import cv2
import mediapipe as mp
import pyttsx3
import numpy as np
import threading
import time
import tkinter as tk
from tkinter import simpledialog, scrolledtext
import json
import os

class EnhancedGestureToSpeech:
    def __init__(self):
        # Initialize MediaPipe Hand Landmark Detection
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize Text-to-Speech Engine
        self.engine = pyttsx3.init()
        
        # Set voice properties
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('rate', 150)  # Speech rate
        if voices:
            self.engine.setProperty('voice', voices[0].id)  # Default voice
        
        self.tts_queue = []
        self.tts_thread = threading.Thread(target=self.run_tts, daemon=True)
        self.tts_thread.start()
        
        # Create config directory if it doesn't exist
        self.config_dir = os.path.join(os.path.expanduser("~"), ".gesture_speech")
        os.makedirs(self.config_dir, exist_ok=True)
        self.config_file = os.path.join(self.config_dir, "gestures.json")
        
        # Load or create default gesture mappings
        self.load_gestures()
        
        # History of spoken phrases
        self.spoken_history = []
        
        # Hand landmark configurations
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Detect up to two hands for more complex gestures
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        
        self.last_gesture = None
        self.last_gesture_time = 0.0
        self.debounce_time = 1.2  # Reduced debounce time for more responsive interaction
        
        # UI elements
        self.display_text = ""
        self.ui_mode = "simple"  # "simple" or "advanced"
        self.show_landmarks = True
        self.current_mode = "speak"  # "speak" or "learn"
        self.learning_gesture = None
        
        # Create settings window
        self.create_settings_window()

    def load_gestures(self):
        # Default gesture mappings with common phrases
        default_gestures = {
            'open_palm': 'Hello, how are you?',
            'peace_sign': 'I am doing well, thank you.',
            'thumbs_up': 'Yes, that sounds good.',
            'pointing': 'Can you help me please?',
            'fist': 'No, I don\'t want that.',
            'ok_sign': 'I need water, please.',
            'pinch': 'I\'m feeling hungry.',
            'telephone': 'Please call for me.',
            'wave': 'I need to sleep.',
            'l_shape': 'I\'m feeling tired.'
        }
        
        # Try to load from config file, or use defaults
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.gestures = json.load(f)
            else:
                self.gestures = default_gestures
                self.save_gestures()
        except Exception as e:
            print(f"Error loading gestures: {e}")
            self.gestures = default_gestures
    
    def save_gestures(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.gestures, f, indent=4)
        except Exception as e:
            print(f"Error saving gestures: {e}")
    
    def create_settings_window(self):
        # Create a separate thread for the settings window
        self.settings_thread = threading.Thread(target=self._run_settings_window, daemon=True)
        self.settings_thread.start()
    
    def _run_settings_window(self):
        self.settings_root = tk.Tk()
        self.settings_root.title("Gesture-to-Speech Settings")
        self.settings_root.geometry("500x600")
        
        # Create tabs
        tab_control = tk.Frame(self.settings_root)
        
        # History tab
        history_frame = tk.LabelFrame(tab_control, text="Speech History")
        history_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.history_text = scrolledtext.ScrolledText(history_frame, wrap=tk.WORD, height=8)
        self.history_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Current speech display
        current_frame = tk.LabelFrame(tab_control, text="Current Speech")
        current_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.current_text = tk.Label(current_frame, text="", font=("Arial", 16), wraplength=450)
        self.current_text.pack(fill="both", expand=True, padx=5, pady=10)
        
        # Gestures tab
        gestures_frame = tk.LabelFrame(tab_control, text="Customize Gestures")
        gestures_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.gesture_listbox = tk.Listbox(gestures_frame, height=8)
        self.gesture_listbox.pack(fill="both", expand=True, padx=5, pady=5)
        self.update_gesture_listbox()
        
        buttons_frame = tk.Frame(gestures_frame)
        buttons_frame.pack(fill="x", padx=5, pady=5)
        
        edit_btn = tk.Button(buttons_frame, text="Edit Phrase", command=self.edit_gesture)
        edit_btn.pack(side=tk.LEFT, padx=5)
        
        learn_btn = tk.Button(buttons_frame, text="Learn New Gesture", command=self.start_learning_mode)
        learn_btn.pack(side=tk.LEFT, padx=5)
        
        # Settings tab
        settings_frame = tk.LabelFrame(tab_control, text="Settings")
        settings_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Voice speed
        speed_frame = tk.Frame(settings_frame)
        speed_frame.pack(fill="x", padx=5, pady=5)
        
        tk.Label(speed_frame, text="Speech Rate:").pack(side=tk.LEFT)
        
        self.speed_var = tk.IntVar(value=150)
        speed_scale = tk.Scale(speed_frame, from_=100, to=200, orient=tk.HORIZONTAL, 
                               variable=self.speed_var, command=self.update_speech_rate)
        speed_scale.pack(side=tk.LEFT, fill="x", expand=True)
        
        # Debounce time
        debounce_frame = tk.Frame(settings_frame)
        debounce_frame.pack(fill="x", padx=5, pady=5)
        
        tk.Label(debounce_frame, text="Gesture Sensitivity:").pack(side=tk.LEFT)
        
        self.debounce_var = tk.DoubleVar(value=self.debounce_time)
        debounce_scale = tk.Scale(debounce_frame, from_=0.5, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, 
                                 variable=self.debounce_var, command=self.update_debounce)
        debounce_scale.pack(side=tk.LEFT, fill="x", expand=True)
        
        # Show landmarks
        self.landmarks_var = tk.BooleanVar(value=self.show_landmarks)
        landmarks_check = tk.Checkbutton(settings_frame, text="Show Hand Landmarks", 
                                        variable=self.landmarks_var, command=self.toggle_landmarks)
        landmarks_check.pack(anchor="w", padx=5, pady=5)
        
        tab_control.pack(fill="both", expand=True)
        
        # Update loop
        self.update_settings_ui()
        self.settings_root.mainloop()
    
    def update_settings_ui(self):
        if hasattr(self, 'settings_root') and self.settings_root.winfo_exists():
            # Update current text
            self.current_text.config(text=self.display_text if self.display_text else "No gesture detected")
            
            # Update history
            self.history_text.delete(1.0, tk.END)
            for text in self.spoken_history[-10:]:  # Show last 10 items
                self.history_text.insert(tk.END, f"• {text}\n")
            
            self.settings_root.after(200, self.update_settings_ui)
    
    def update_gesture_listbox(self):
        self.gesture_listbox.delete(0, tk.END)
        for gesture, phrase in self.gestures.items():
            self.gesture_listbox.insert(tk.END, f"{gesture}: {phrase}")
    
    def edit_gesture(self):
        selection = self.gesture_listbox.curselection()
        if not selection:
            return
            
        selected = self.gesture_listbox.get(selection[0])
        gesture = selected.split(":")[0].strip()
        
        new_phrase = simpledialog.askstring("Edit Phrase", 
                                          f"Enter new phrase for gesture '{gesture}':",
                                          initialvalue=self.gestures.get(gesture, ""))
        
        if new_phrase:
            self.gestures[gesture] = new_phrase
            self.save_gestures()
            self.update_gesture_listbox()
    
    def start_learning_mode(self):
        gesture_name = simpledialog.askstring("New Gesture", 
                                           "Enter a name for the new gesture:")
        if not gesture_name:
            return
            
        phrase = simpledialog.askstring("New Gesture", 
                                     f"Enter the phrase for gesture '{gesture_name}':")
        if not phrase:
            return
        
        self.learning_gesture = gesture_name
        self.gestures[gesture_name] = phrase
        self.current_mode = "learn"
        
        # Show instructions on main window
        self.display_text = f"LEARNING MODE: Make the gesture for '{gesture_name}' and hold for 3 seconds"
    
    def update_speech_rate(self, value):
        self.engine.setProperty('rate', int(value))
    
    def update_debounce(self, value):
        self.debounce_time = float(value)
    
    def toggle_landmarks(self):
        self.show_landmarks = self.landmarks_var.get()
    
    def run_tts(self):
        while True:
            if self.tts_queue:
                text = self.tts_queue.pop(0)
                self.engine.say(text)
                self.engine.runAndWait()
            time.sleep(0.1)
    
    def detect_gesture(self, landmarks):
        """
        Classify hand gestures based on landmark positions
        """
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = landmarks[self.mp_hands.HandLandmark.PINKY_PIP]
        
        # Open palm detection (All fingers extended and apart)
        if (all(finger.y < wrist.y for finger in [index_tip, middle_tip, ring_tip, pinky_tip]) and 
            abs(thumb_tip.x - index_tip.x) > 0.04):
            return 'open_palm'
        
        # Fist detection (All fingers curled)
        if (all(tip.y > pip.y for tip, pip in [
            (index_tip, index_pip), 
            (middle_tip, middle_pip), 
            (ring_tip, ring_pip), 
            (pinky_tip, pinky_pip)
        ])):
            return 'fist'
        
        # Peace sign detection (Index and middle extended, others curled)
        if (index_tip.y < index_pip.y and 
            middle_tip.y < middle_pip.y and 
            ring_tip.y > ring_pip.y and 
            pinky_tip.y > pinky_pip.y):
            return 'peace_sign'
        
        # Thumbs up detection (Thumb extended upwards, others curled)
        if (thumb_tip.y < thumb_ip.y and
            all(tip.y > pip.y for tip, pip in [
                (index_tip, index_pip), 
                (middle_tip, middle_pip), 
                (ring_tip, ring_pip), 
                (pinky_tip, pinky_pip)
            ])):
            return 'thumbs_up'
        
        # OK sign (Thumb and index tips touching, forming a circle)
        if (abs(thumb_tip.x - index_tip.x) < 0.05 and 
            abs(thumb_tip.y - index_tip.y) < 0.05):
            return 'ok_sign'
        
        # Pinch gesture (Thumb and index pinching)
        if (abs(thumb_tip.x - index_tip.x) < 0.03 and 
            abs(thumb_tip.y - index_tip.y) < 0.03 and
            middle_tip.y < middle_pip.y):
            return 'pinch'
        
        # Telephone gesture (Thumb and pinky extended, others curled)
        if (thumb_tip.y < thumb_ip.y and 
            pinky_tip.y < pinky_pip.y and
            index_tip.y > index_pip.y and 
            middle_tip.y > middle_pip.y and 
            ring_tip.y > ring_pip.y):
            return 'telephone'
        
        # Wave gesture (All fingers extended and together moving side to side)
        if all(tip.y < pip.y for tip, pip in [
                (index_tip, index_pip), 
                (middle_tip, middle_pip), 
                (ring_tip, ring_pip), 
                (pinky_tip, pinky_pip)
            ]) and abs(index_tip.x - pinky_tip.x) < 0.1:
            return 'wave'
        
        # L shape (Thumb and index extended forming an L, others curled)
        if (thumb_tip.x < index_tip.x and
            thumb_tip.y < thumb_ip.y and
            index_tip.y < index_pip.y and
            middle_tip.y > middle_pip.y and
            ring_tip.y > ring_pip.y and
            pinky_tip.y > pinky_pip.y):
            return 'l_shape'
        
        return None
    
    def run(self):
        """
        Main method to capture video and convert gestures to speech
        """
        cap = cv2.VideoCapture(0)
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Create a window that can be resized
        cv2.namedWindow('Gesture to Speech', cv2.WINDOW_NORMAL)
        
        # Learning mode variables
        learning_start_time = 0
        learning_gesture_detected = False
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to capture video")
                break
            
            # Flip and convert image
            image = cv2.flip(image, 1)
            
            # Create a working copy of the image
            display_image = image.copy()
            
            # Add UI elements
            self.add_ui_elements(display_image)
            
            # Process hand landmarks
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks if enabled
                    if self.show_landmarks:
                        self.mp_drawing.draw_landmarks(
                            display_image, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
                    
                    # Detect gesture
                    gesture = self.detect_gesture(hand_landmarks.landmark)
                    
                    if self.current_mode == "learn" and gesture is None:
                        # We're in learning mode and waiting for a stable pose
                        if not learning_gesture_detected:
                            learning_start_time = time.time()
                            learning_gesture_detected = True
                        
                        # If we've held the pose for 3 seconds, save it
                        if time.time() - learning_start_time > 3:
                            self.save_gestures()
                            self.update_gesture_listbox()
                            
                            # Exit learning mode
                            self.current_mode = "speak"
                            self.display_text = f"New gesture '{self.learning_gesture}' learned!"
                            self.tts_queue.append(f"New gesture learned.")
                            
                            self.learning_gesture = None
                            learning_gesture_detected = False
                    
                    elif self.current_mode == "speak" and gesture:
                        # Normal speaking mode
                        if gesture != self.last_gesture and (time.time() - self.last_gesture_time) > self.debounce_time:
                            if gesture in self.gestures:
                                phrase = self.gestures[gesture]
                                self.display_text = f"{gesture}: {phrase}"
                                self.tts_queue.append(phrase)
                                self.spoken_history.append(phrase)
                                self.last_gesture = gesture
                                self.last_gesture_time = time.time()
            
            # Display the image
            cv2.imshow('Gesture to Speech', display_image)
            
            # Exit on 'q' key press
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.toggle_ui_mode()
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Close the settings window if it exists
        if hasattr(self, 'settings_root') and self.settings_root.winfo_exists():
            self.settings_root.destroy()
    
    def add_ui_elements(self, image):
        """Add UI elements to the display image"""
        h, w = image.shape[:2]
        
        # Add semi-transparent overlay for text background at the bottom
        overlay = image.copy()
        cv2.rectangle(overlay, (0, h-150), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        # Add mode indicator
        mode_text = f"MODE: {'LEARNING' if self.current_mode == 'learn' else 'SPEAKING'}"
        cv2.putText(image, mode_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Add current speech text
        if self.display_text:
            # Split text for multiple lines
            words = self.display_text.split()
            lines = []
            current_line = ""
            
            for word in words:
                if len(current_line + " " + word) > 50:  # Max chars per line
                    lines.append(current_line)
                    current_line = word
                else:
                    if current_line:
                        current_line += " " + word
                    else:
                        current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Draw text lines
            y_pos = h - 100
            for line in lines:
                cv2.putText(image, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                y_pos += 40
        
        # Add gesture history
        if self.ui_mode == "advanced" and self.spoken_history:
            y_pos = 80
            cv2.putText(image, "Recent phrases:", (w-300, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_pos += 30
            
            for phrase in self.spoken_history[-3:]:  # Show last 3 items
                if len(phrase) > 25:
                    phrase = phrase[:22] + "..."
                cv2.putText(image, f"• {phrase}", (w-300, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                y_pos += 30
        
        # Add help text
        cv2.putText(image, "Press 'q' to quit, 's' to change view", (10, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    def toggle_ui_mode(self):
        if self.ui_mode == "simple":
            self.ui_mode = "advanced"
        else:
            self.ui_mode = "simple"

if __name__ == "__main__":
    converter = EnhancedGestureToSpeech()
    converter.run()