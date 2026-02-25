import os
import random
import torch
import torchaudio
from tqdm import tqdm

DATA_DIR = r"C:\Users\ADMIN\.cache\huggingface\datasets\downloads\extracted\9a7e9bf17031c538867f3c80e35553fad8bc332a9f0647a4cdd8c97b9bb0e910\train"
TARGET_SAMPLE_RATE = 16000
DEGRADE_PERCENTAGE = 0.30
SEED = 42

random.seed(SEED)

audio_extensions = [".wav", ".flac", ".mp3"]
audio_files = []

for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(os.path.join(root, file))

print(f"Total audio files found: {len(audio_files)}")

num_to_degrade = int(len(audio_files) * DEGRADE_PERCENTAGE)
files_to_degrade = random.sample(audio_files, num_to_degrade)

print(f"Degrading {num_to_degrade} files...")

for file_path in tqdm(files_to_degrade):
    waveform, sample_rate = torchaudio.load(file_path)

    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)

    # 16k → 8k
    downsampler = torchaudio.transforms.Resample(TARGET_SAMPLE_RATE, 8000)
    degraded = downsampler(waveform)

    # 8k → 16k
    upsampler = torchaudio.transforms.Resample(8000, TARGET_SAMPLE_RATE)
    restored = upsampler(degraded)

    # Overwrite
    torchaudio.save(file_path, restored, TARGET_SAMPLE_RATE)

print("Degradation complete ✅")