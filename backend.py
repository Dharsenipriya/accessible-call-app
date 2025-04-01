import pyttsx3  

def speak_text(text):
    engine = pyttsx3.init()
    
    # Set properties (speed, volume, voice)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

    # Speak the text
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    while True:
        text = input("Enter text to speak (or 'exit' to quit): ")
        if text.lower() == "exit":
            break
        speak_text(text)
