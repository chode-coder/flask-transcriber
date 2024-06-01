import whisper
import torch
import numpy as np
import soundfile as sf

# Load the Whisper model
model = whisper.load_model("base")

# Path to the audio file
audio_file_path = "output/segment_000.wav"

# Function to split audio into chunks
def split_audio(audio, chunk_size):
    return [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

# Transcribe the audio file
print(f"Transcribing {audio_file_path}...")

# Load and preprocess the audio
try:
    # Load the audio file
    audio, sr = sf.read(audio_file_path)
    print(f"Loaded audio file with sample rate {sr} and shape {audio.shape}")
    
    # Check if the sample rate is correct
    assert sr == 16000, f"Sample rate should be 16000, but got {sr}"
    
    # Convert the audio to float32
    audio = audio.astype(np.float32)
    
    # Split audio into 30-second chunks (480000 samples)
    chunk_size = 480000
    audio_chunks = split_audio(audio, chunk_size)
    
    transcription = ""
    
    # Process each chunk
    for i, chunk in enumerate(audio_chunks):
        print(f"Processing chunk {i + 1}/{len(audio_chunks)}")
        
        # Pad the chunk if it's shorter than the chunk size
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
        
        # Convert to tensor and add batch dimension
        audio_tensor = torch.tensor(chunk).unsqueeze(0).to(model.device)
        print(f"Audio tensor shape for chunk {i + 1}: {audio_tensor.shape}")
        
        # Create log-mel spectrogram
        mel = whisper.log_mel_spectrogram(audio_tensor)
        print(f"Mel spectrogram shape for chunk {i + 1}: {mel.shape}")
        
        # Decode the audio with adjusted sensitivity
        options = whisper.DecodingOptions(fp16=False, temperature=0.0)
        result = model.decode(mel, options)
        
        # Extract the text from the DecodingResult objects in the list
        chunk_transcription = " ".join([segment.text for segment in result])
        transcription += chunk_transcription + " "
        
        # Log the no_speech_prob value
        for segment in result:
            print(f"Segment no_speech_prob: {segment.no_speech_prob}, Text: {segment.text}")
    
    # Print the complete transcription
    print(f"Transcription of {audio_file_path}:")
    print(transcription.strip())
except Exception as e:
    print(f"Error transcribing {audio_file_path}: {e}")
