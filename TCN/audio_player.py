"""
Simple audio player for prediction results
"""

import soundfile as sf
import sounddevice as sd
import os


def play_audio(prediction_class, audio_folder="../AudioFile"):
    """Play audio based on prediction class"""
    
    # Map prediction classes to audio files
    audio_files = {
        0: "Correct.mp3",
        1: "Error0.mp3", 
        2: "Error1.mp3",
        3: "Error2.mp3",
        4: "Error3.mp3",
        5: "Error4.mp3",
        6: "Error5.mp3"
    }
    
    # Get audio file for this prediction
    if prediction_class not in audio_files:
        print(f"No audio file for prediction {prediction_class}")
        return
    
    filename = audio_files[prediction_class]
    filepath = os.path.join(audio_folder, filename)
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Audio file not found: {filepath}")
        return
    
    try:
        # Load and play audio
        audio, sr = sf.read(filepath, dtype="float32")
        
        # Handle mono/stereo
        if audio.ndim == 1:
            audio_out = audio
        else:
            audio_out = audio[:, 0]  # Use first channel
        
        sd.play(audio_out, sr)
        sd.wait()
        
    except Exception as e:
        print(f"Error playing audio: {e}")