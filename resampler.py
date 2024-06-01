import soundfile as sf
import resampy

# Path to the original audio file
original_audio_path = "audio/call_0d1b75a1ccc148ddaac5a89cea9daabd_1717189593.mp3"
# Path to save the resampled audio file
resampled_audio_path = "audio/call_0d1b75a1ccc148ddaac5a89cea9daabd_1717189593_resampled.wav"

# Load the original audio file
audio, sr = sf.read(original_audio_path)
print(f"Original sample rate: {sr}")

# Resample to 16000 Hz if needed
if sr != 16000:
    audio = resampy.resample(audio, sr, 16000)
    sr = 16000
    print(f"Resampled audio to {sr} Hz")

# Save the resampled audio file
sf.write(resampled_audio_path, audio, sr)
print(f"Resampled audio saved to: {resampled_audio_path}")
