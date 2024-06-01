import whisper
import torch
import numpy as np
import soundfile as sf
import os

# Load the Whisper model
model = whisper.load_model("base")

# Path to the resampled audio file
audio_file_path = "audio/call_0d1b75a1ccc148ddaac5a89cea9daabd_1717189593_resampled.wav"

# Output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Function to split audio into chunks
def split_audio(audio, chunk_size):
    return [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

# Function to transcribe a single chunk
def transcribe_chunk(chunk, index):
    try:
        # Pad the chunk if it's shorter than the chunk size
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
        
        # Convert to tensor and add batch dimension
        audio_tensor = torch.tensor(chunk).unsqueeze(0).to(model.device)
        print(f"Audio tensor shape for chunk {index}: {audio_tensor.shape}")
        
        # Create log-mel spectrogram
        mel = whisper.log_mel_spectrogram(audio_tensor)
        print(f"Mel spectrogram shape for chunk {index}: {mel.shape}")
        
        # Ensure mel spectrogram has the correct dimensions
        if mel.shape[2] != 3000:
            mel = torch.nn.functional.pad(mel, (0, 3000 - mel.shape[2]), mode='constant', value=0)
            print(f"Adjusted mel spectrogram shape for chunk {index}: {mel.shape}")
        
        # Decode the audio
        options = whisper.DecodingOptions(fp16=False, temperature=0.0)
        result = model.decode(mel, options)
        
        # Extract the text from the DecodingResult objects in the list
        if isinstance(result, list):
            transcription = " ".join([segment.text for segment in result])
        else:
            transcription = result.text
        
        return transcription
    except Exception as e:
        print(f"Error transcribing chunk {index}: {e}")
        return ""

# Load and preprocess the audio file
audio, sr = sf.read(audio_file_path)
print(f"Sample rate: {sr}")

# Convert the audio to float32
audio = audio.astype(np.float32)

# Split audio into 30-second chunks (480000 samples)
chunk_size = 480000
audio_chunks = split_audio(audio, chunk_size)

# Transcribe each chunk sequentially
transcriptions = []
for i, chunk in enumerate(audio_chunks):
    transcription = transcribe_chunk(chunk, i)
    transcriptions.append(transcription)

# Combine transcriptions
merged_transcription = "\n\n".join(transcriptions)

# Save the merged transcription
with open(os.path.join(output_dir, "merged_transcript.txt"), "w") as f:
    f.write(merged_transcription)

print("Merged transcript saved to:", os.path.join(output_dir, "merged_transcript.txt"))
