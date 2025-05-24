import os
import torch
import torchaudio
import torch.nn as nn
from torchaudio.transforms import Resample, MelSpectrogram
from tqdm import tqdm
import numpy as np

class AudioTransformerPreprocessor:
    def __init__(self, 
                 target_sr=16000,
                 n_mels=128,
                 n_fft=400,
                 win_length=400,
                 hop_length=160,
                 target_length=300):
        """
        Args:
            target_sr: Target sample rate (16000 Hz)
            n_mels: Number of Mel bands (128)
            n_fft: FFT window size (400)
            win_length: Window length (400)
            hop_length: Hop length (160)
            target_length: Fixed number of time steps (300)
        """
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.target_length = target_length
        
        self.mel_transform = MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

    def preprocess_audio(self, waveform, sample_rate):
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        if sample_rate != self.target_sr:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.target_sr)
            waveform = resampler(waveform)
            
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)
        
        mel_spec = self.mel_transform(waveform)
        
        mel_spec = torch.log(mel_spec + 1e-9)
        
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
        
        current_length = mel_spec.shape[-1]
        if current_length < self.target_length:
            pad_amount = self.target_length - current_length
            mel_spec = nn.functional.pad(mel_spec, (0, pad_amount))
        else:
            mel_spec = mel_spec[:, :, :self.target_length]
            
        mel_spec = mel_spec.squeeze(0).transpose(0, 1)  
        
        return mel_spec

    def process_directory(self, input_dir, output_dir):
        """Process all .wav files in a directory"""
        os.makedirs(output_dir, exist_ok=True)
        audio_files = []
        
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith('.wav'):
                    audio_files.append(os.path.join(root, file))
        
        if not audio_files:
            print(f"No WAV files found in {input_dir}!")
            return
        
        print(f"Found {len(audio_files)} files to process")
        
        for filepath in tqdm(audio_files, desc="Processing audio"):
            try:
                waveform, sample_rate = torchaudio.load(filepath)
                processed = self.preprocess_audio(waveform, sample_rate)
                
                rel_path = os.path.relpath(filepath, start=input_dir)
                output_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".pt")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torch.save(processed, output_path)
                
            except Exception as e:
                print(f"\nError processing {filepath}: {str(e)}")
                continue

if __name__ == "__main__":
    preprocessor = AudioTransformerPreprocessor(
        target_sr=16000,
        n_mels=128,
        n_fft=400,
        win_length=400,
        hop_length=160,
        target_length=300
    )
    
    input_directory = "path to data directory"  
    output_directory = "path to output directory"
    
    preprocessor.process_directory(input_directory, output_directory)
    print("\nPreprocessing complete!")


sample = torch.load("preprocessed_spectrograms/example.pt")
print(sample.shape)  

model_input = sample.unsqueeze(0)  
