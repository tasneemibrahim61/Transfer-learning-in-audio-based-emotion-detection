import os
import torch
import torchaudio
from torchaudio.transforms import Resample
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from pathlib import Path

class AudioDatasetPreprocessor:
    def __init__(self, target_sr=16000, duration=4.0, n_workers=4):
        self.target_sr = target_sr
        self.target_len = int(target_sr * duration)
        self.n_workers = n_workers
        self.resamplers = {}  

    def _get_resampler(self, orig_sr):
        if orig_sr not in self.resamplers:
            self.resamplers[orig_sr] = Resample(orig_sr, self.target_sr)
        return self.resamplers[orig_sr]

    def _process_file(self, args):
        filepath, output_dir = args
        try:
            waveform, sr = torchaudio.load(filepath)
            waveform = waveform.mean(dim=0)  
            
            if sr != self.target_sr:
                waveform = self._get_resampler(sr)(waveform)
            
            if len(waveform) > self.target_len:
                waveform = waveform[:self.target_len]
            else:
                pad = self.target_len - len(waveform)
                waveform = torch.nn.functional.pad(waveform, (0, pad))
            
            waveform = self._trim_silence(waveform)  

            waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-5)
            
            rel_path = os.path.relpath(filepath, start=os.path.dirname(output_dir))
            output_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".pt")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(waveform, output_path)
            
            return True
        except Exception as e:
            print(f"\nError processing {filepath}: {str(e)}")
            return False

    def process_directory(self, input_dir, output_dir):
        input_dir = os.path.abspath(input_dir)
        output_dir = os.path.abspath(output_dir)
        
        print(f"\nInput directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        
        files = []
        for root, _, filenames in os.walk(input_dir):
            for f in filenames:
                if f.lower().endswith('.wav'):
                    files.append(os.path.join(root, f))
        
        print(f"\nFound {len(files)} .wav files in directory tree:")
        for f in files[:3]: 
            print(f"  {f}")
        if len(files) > 3:
            print(f"  ... and {len(files)-3} more files")
        
        if not files:
            print("\nERROR: No WAV files found. Please check:")
            print("1. The input path is correct")
            print("2. Files have .wav extension (case insensitive)")
            print("3. Files exist in the directory/subdirectories")
            print(f"\nCurrent directory contents: {os.listdir(input_dir)}")
            return 0
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nStarting processing...")
        with Pool(self.n_workers) as pool:
            args = [(f, output_dir) for f in files]
            results = list(tqdm(
                pool.imap(self._process_file, args),
                total=len(files),
                desc="Processing",
                unit="file"
            ))
        
        success_count = sum(results)
        print(f"\nSuccessfully processed {success_count}/{len(files)} files")
        if success_count < len(files):
            print(f"Failed to process {len(files)-success_count} files")
        return success_count

if __name__ == "__main__":
    config = {
        "input_dir": "sampledata/english/Espeaker1",  
        "output_dir": "preprocessed",
        "target_sr": 16000,       
        "duration": 4.0,            
        "n_workers": 4              
    }
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config["input_dir"] = os.path.join(script_dir, config["path to data directory"])
    config["output_dir"] = os.path.join(script_dir, config["path to output data directory"])
    
    preprocessor = AudioDatasetPreprocessor(
        target_sr=config["target_sr"],
        duration=config["duration"],
        n_workers=config["n_workers"]
    )
    
    preprocessor.process_directory(config["input_dir"], config["output_dir"])

sample = torch.load("preprocessed/sampledata/english/Espeaker1/Angry/0011_000379.pt")
print(sample.shape) 
print(sample.dtype) 
print(f"Mean: {sample.mean():.4f}, Std: {sample.std():.4f}")  
