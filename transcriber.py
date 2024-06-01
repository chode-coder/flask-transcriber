import whisper
import torch
import numpy as np
import soundfile as sf
import os

# Load the Whisper model
model = whisper.load_model("base")

def transcribe(audio_file_path):
    # Your transcription logic here
    # For example:
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess the audio file
    audio, sr = sf.read(audio_file_path)
    audio = audio.astype(np.float32)
    
    # Split audio into 30-second chunks (480000 samples)
    chunk_size = 480000
    audio_chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

    # Transcribe each chunk sequentially
    transcriptions = []
    for i, chunk in enumerate(audio_chunks):
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
        audio_tensor = torch.tensor(chunk).unsqueeze(0).to(model.device)
        mel = whisper.log_mel_spectrogram(audio_tensor)
        if mel.shape[2] != 3000:
            mel = torch.nn.functional.pad(mel, (0, 3000 - mel.shape[2]), mode='constant', value=0)
        options = whisper.DecodingOptions(fp16=False, temperature=0.0)
        result = model.decode(mel, options)
        transcription = result.text if isinstance(result, whisper.DecodingResult) else " ".join([segment.text for segment in result])
        transcriptions.append(transcription)
    
    merged_transcription = "\n".join(transcriptions)
    with open(os.path.join(output_dir, "merged_transcript.txt"), "w") as f:
        f.write(merged_transcription)
    print("Merged transcript saved to:", os.path.join(output_dir, "merged_transcript.txt"))
