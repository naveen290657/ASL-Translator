import queue
import threading
import sys
import os
import traceback

# Thread-safe Speech Queue
_speech_queue = queue.Queue()
_worker_thread = None
_stop_event = threading.Event()

def _tts_worker():
    """Worker thread that initializes COM and processes speech requests sequentially.

    IMPORTANT: pyttsx3's SAPI5 driver on Windows has a well-known bug where reusing
    a single engine.init() instance across multiple engine.runAndWait() calls works
    for the *first* utterance and then silently goes dead for every call after that.
    Since this app calls speak() many times in a session (once per recognized sign),
    a long-lived shared engine would produce exactly that symptom: audio works once,
    then never again. The reliable fix is to create a brand-new engine for each
    utterance and dispose of it right after.
    """
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except Exception as e:
        print(f"[TTS] pythoncom.CoInitialize() failed (pywin32 missing/broken?): {e}")

    while not _stop_event.is_set():
        try:
            text = _speech_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if text is None:
            break

        print(f"[TTS] Speaking: '{text}'")
        spoke = False
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            if not voices:
                print("[TTS] WARNING: pyttsx3 reports zero installed voices. "
                      "Windows has no SAPI5 voice registered, so nothing can play.")
            engine.setProperty('rate', 160)     # Natural speaking rate
            engine.setProperty('volume', 1.0)   # Max volume
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
            spoke = True
            print(f"[TTS] pyttsx3 finished speaking '{text}'")
        except Exception:
            print("[TTS] pyttsx3 raised an exception, falling back to PowerShell:")
            traceback.print_exc()
            spoke = False

        if not spoke:
            _speak_system_fallback(text)

        _speech_queue.task_done()

def _speak_system_fallback(text):
    """Fallback using Windows PowerShell SpeechSynthesizer if pyttsx3 encounters issues.

    NOTE: this only works on Windows. On macOS/Linux this will fail (no 'powershell'
    binary), which is itself a likely cause of total silence if you're not on Windows.
    """
    try:
        clean_text = text.replace('"', '').replace("'", '')
        cmd = f'powershell -c "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{clean_text}\')"'
        ret = os.system(cmd)
        if ret != 0:
            print(f"[TTS] PowerShell fallback returned exit code {ret} "
                  f"(non-zero usually means it failed — are you on Windows?)")
        else:
            print(f"[TTS] PowerShell fallback spoke '{text}'")
    except Exception:
        print("[TTS] PowerShell fallback also raised an exception:")
        traceback.print_exc()

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
    print("Testing TTS worker in isolation (no camera, no Streamlit)...")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    try:
        import pyttsx3
        print(f"pyttsx3 version: {pyttsx3.__version__ if hasattr(pyttsx3, '__version__') else 'unknown'}")
    except ImportError:
        print("pyttsx3 is NOT installed. Run: pip install pyttsx3")
    speak("Hello, ASL Recognition is active.")
    speak("This is a second sentence to confirm repeated speech works.")
    print("Waiting for both utterances to finish (no fixed timeout this time)...")
    _speech_queue.join()  # blocks until both queued items are actually processed
    print("TTS test completed. If you heard both sentences above, repeated speech works correctly.")