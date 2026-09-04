import queue
import threading
import sys
import os

# Thread-safe Speech Queue
_speech_queue = queue.Queue()
_worker_thread = None
_stop_event = threading.Event()

def _tts_worker():
    """Worker thread that initializes COM and processes speech requests sequentially."""
    engine = None
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except Exception:
        pass

    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)     # Natural speaking rate
        engine.setProperty('volume', 1.0)   # Max volume
    except Exception:
        engine = None

    while not _stop_event.is_set():
        try:
            text = _speech_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if text is None:
            break

        if engine is not None:
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                _speak_system_fallback(text)
        else:
            _speak_system_fallback(text)

        _speech_queue.task_done()

def _speak_system_fallback(text):
    """Fallback using Windows PowerShell SpeechSynthesizer if pyttsx3 encounters issues."""
    try:
        clean_text = text.replace('"', '').replace("'", '')
        cmd = f'powershell -c "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{clean_text}\')"'
        os.system(cmd)
    except Exception:
        pass

def init_tts():
    """Start the background speech worker thread."""
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _stop_event.clear()
        _worker_thread = threading.Thread(target=_tts_worker, daemon=True, name="TTS-Worker")
        _worker_thread.start()

def speak(text):
    """Queue text to be spoken asynchronously without blocking the caller."""
    if not text or not str(text).strip():
        return
    init_tts()
    _speech_queue.put(str(text).strip())

# Automatically initialize worker
init_tts()

if __name__ == '__main__':
    print("Testing TTS worker...")
    speak("Hello, ASL Recognition is active.")
    import time
    time.sleep(2)
    print("TTS test completed successfully.")
