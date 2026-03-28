"""Multimodal mood detection backend for Streamlit app.

Combines voice transcription + emotion detection from webcam.
"""

import time
import threading
from typing import Optional, Dict, Any
import numpy as np

try:
    import sounddevice as sd
    import speech_recognition as sr
    _HAS_VOICE = True
except ImportError:
    _HAS_VOICE = False

from src.emotion_detection import detect_mood_from_frame

def detect_multimodal_mood_backend(
    voice_duration_sec: float = 2.0,
    use_parallel_threads: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Backend mood + voice detection for Streamlit private scanning.
    
    Returns dict with mood, voice transcript, source, top_moods.
    """
    result = {
        "mood": "calm",
        "spoken_text": "",
        "source": "neutral",
        "top_moods": [("calm", 0.8)],
        "voice_transcript_error": None,
    }
    
    mood_from_face = None
    voice_text = ""
    
    # Thread for face detection (mock webcam - use latest frame or None for now)
    def detect_face_mood():
        nonlocal mood_from_face
        # In real impl, grab latest frame from st.camera or session state
        frame = None  # Placeholder - integrate with Streamlit camera
        if frame is not None:
            mood_from_face = detect_mood_from_frame(frame)
    
    # Thread for voice (if available)
    voice_error = None
    if _HAS_VOICE:
        def capture_voice():
            nonlocal voice_text, voice_error
            try:
                r = sr.Recognizer()
                with sd.InputStream(callback=None, channels=1, samplerate=16000) as stream:
                    audio_data = sd.rec(int(voice_duration_sec * 16000), samplerate=16000, channels=1)
                    sd.wait()
                    audio = sr.AudioData(audio_data.tobytes(), 16000, 2)
                    voice_text = r.recognize_google(audio)
            except Exception as e:
                voice_error = str(e)
        
        voice_thread = threading.Thread(target=capture_voice, daemon=True)
        voice_thread.start()
        voice_thread.join(timeout=voice_duration_sec + 1)
    
    # Mock face thread (parallel)
    if use_parallel_threads:
        face_thread = threading.Thread(target=detect_face_mood, daemon=True)
        face_thread.start()
        face_thread.join(timeout=1.0)
    
    result["spoken_text"] = voice_text.strip()
    if voice_error:
        result["voice_transcript_error"] = voice_error
    
    # Mood fusion logic
    mood_scores = {"calm": 0.5}
    if mood_from_face:
        mood_scores[mood_from_face] = 0.8
    if voice_text:
        # Simple keyword mood from voice
        low = voice_text.lower()
        if any(w in low for w in ["happy", "great", "love"]):
            mood_scores["happy"] = 0.7
        elif any(w in low for w in ["sad", "boring"]):
            mood_scores["sad"] = 0.7
        elif any(w in low for w in ["angry", "hate"]):
            mood_scores["angry"] = 0.7
    
    # Pick top mood
    top_mood = max(mood_scores.items(), key=lambda x: x[1])
    result["mood"] = top_mood[0]
    result["top_moods"] = sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    result["source"] = "voice+face" if mood_from_face and voice_text else ("voice" if voice_text else "face")
    
    return result

