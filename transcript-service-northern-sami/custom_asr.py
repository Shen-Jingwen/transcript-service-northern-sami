import numpy as np
from transformers import pipeline
from librosa import resample
import whisperx
import torch
import tempfile
import soundfile as sf
import os
import threading
import queue
from utils.forced_alignment import generate_word_timestamps
from utils.punctuation_restorer import PunctuationRestorer

ASR_MODELS = [ 
    "GetmanY1/wav2vec2-base-sami-cont-pt-22k-finetuned",
    "GetmanY1/wav2vec2-large-sami-cont-pt-22k-finetuned",
    # "facebook/wav2vec2-base-960h"
]

class CustomAutomaticSpeechRecognizer:
    def __init__(self, model_name=ASR_MODELS[1], device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_name = None  
        self.pipe = None
        self.punctuation_restorer = PunctuationRestorer()
        self._stop_event = threading.Event()  # stop process last audio once upload new audios
        self._current_process = None
        self._result_queue = queue.Queue()
        
        self.chunk_length_s = 10  # Length of audio chunks to process
        self.stream_chunk_s = 9
        self.change_model(model_name)
        
    def change_model(self, model_name):
        if model_name == self.model_name: 
            print(f"Model already set to {model_name}, skipping reload")
            return
        print(f"Changing model from {self.model_name} to {model_name}")
        
        self._stop_event.set()
        if self._current_process and self._current_process.is_alive():
            self._current_process.join(timeout=1.0)
            if self._current_process.is_alive():
                self._current_process.terminate()
            
        if self.pipe is not None:
            del self.pipe
            torch.cuda.empty_cache()
            
        self.model_name = model_name
        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=self.device,
                return_timestamps='word',
                torch_dtype=torch.float32 
            )
            print(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            raise
        self._stop_event.clear()

    def preprocess_audio(self, audio, sr):
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # Convert to float32 if needed
        if audio.dtype != "float32":
            audio = audio.astype(np.float32)
        # Resample if sampling rate is not 16kHz
        if sr != 16000:
            audio = resample(audio, orig_sr=sr, target_sr=16000)
        return audio
          
    def transcribe(self, stream, result_queue):
        sr, audio = stream
        audio = self.preprocess_audio(audio, sr)
        if stream is None:
            raise ValueError("The 'stream' parameter cannot be None.")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            sf.write(temp_audio_file.name, audio, 16000)
            temp_file_name = temp_audio_file.name

        text = ""
        last_chunk_words = []
        min_overlap_words = 3 #remove overlap only when overlap >= min_overlap_words (set to 1 if no need)
        check_overlap_words = 10 #number of words to check overlap for each new chunk
        timestamps = []
        global_offset = 0.0
        last_chunk_end = 0.0
        overlap_ratio = 0.1
        
        for chunk_result in self._streaming_transcribe(audio, sr):
            if self._stop_event.is_set():
                os.remove(temp_file_name)
                return
            chunk_text = chunk_result["text"].strip()
            current_words = chunk_text.split()
            overlap = 0
            
            # handle overlap part caused by chunk process
            if last_chunk_words:
                max_possible_overlap = min(len(last_chunk_words), len(current_words), 10)
                for i in range(max_possible_overlap, 0, -1):
                    last_part = " ".join(last_chunk_words[-i:]).lower().strip(".,!?")
                    current_part = " ".join(current_words[:i]).lower().strip(".,!?")
                    if current_part.startswith(last_part):
                        overlap = i
                        if not current_part == last_part:
                            # the last word in last chunk is not complete
                            overlap = i-1
                            text = " ".join(text.split()[:-1])
                        break
                    else:
                        # the last word in last chunk is rubbish (not complete and split to 2 words)
                        last_part_2 = " ".join(last_chunk_words[-i-1:-1]).lower().strip(".,!?")
                        if current_part.startswith(last_part_2):
                            overlap = i
                            text = " ".join(text.split()[:-1])
                            break
                        else:
                            # the first word in current chunk is rubbish
                            current_part_2 = " ".join(current_words[1:i+1]).lower().strip(".,!?")
                            if current_part_2.startswith(last_part):
                                overlap = i-1
                                current_words = current_words[1:]
                                text = " ".join(text.split()[:-1])
                                break
                    
            
            if overlap >= min_overlap_words:
                # remove overlap directly
                chunk_text = " ".join(current_words[overlap:])
            
            if chunk_text:
                text += " " + chunk_text
                last_chunk_words = current_words[-check_overlap_words:]
            
            temp_timestamps = []
            if "chunks" in chunk_result:
                for idx, ts in enumerate(chunk_result["chunks"]):
                    if overlap > 0 and idx < overlap:
                        continue
                    adjusted_ts = {
                        "text": ts["text"],
                        "start": ts["timestamp"][0] + global_offset,
                        "end": ts["timestamp"][1] + global_offset
                    }
                    temp_timestamps.append(adjusted_ts)
                
                if chunk_result["chunks"]:
                    last_chunk_end = chunk_result["chunks"][-1]["timestamp"][1]
                    global_offset += last_chunk_end - (self.chunk_length_s * overlap_ratio)
            
            # current_text = self.punctuation_restorer.restore(text) ##punctuation, need better solution
            current_text = text
            result_queue.put((current_text, stream, current_text, temp_timestamps))
    
        try:
            final_timestamps = generate_word_timestamps(temp_file_name, text)
        except Exception as e:
            print(f"Error generating timestamps: {e}")
            final_timestamps = []
        
        # final_text = self.punctuation_restorer.restore(text) ##punctuation, need better solution
        final_text = text
        os.remove(temp_file_name)
        result_queue.put((final_text, stream, final_text, final_timestamps))
            
              
    def _streaming_transcribe(self, audio, sr):
        """Generator that yields partial transcription results"""
        audio_list = audio.tolist()     
        # Process audio in chunks
        chunk_size = int(self.chunk_length_s * sr)
        stream_chunk_size = int(self.stream_chunk_s * sr)
        overlap_ratio = 0.1
        
        for i in range(0, len(audio_list), stream_chunk_size):
            if self._stop_event.is_set():
                return
            start = max(0, i - int(overlap_ratio * stream_chunk_size)) 
            end = min(len(audio_list), start + chunk_size)
            
            chunk = audio_list[start:end]
            
            # Pad last chunk if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                
            chunk_np = np.array(chunk, dtype=np.float32)
            # Process chunk
            output = self.pipe(
                chunk_np,
                chunk_length_s=self.chunk_length_s,
                stride_length_s=(overlap_ratio, 0.1),
                return_timestamps='word'
            )
            print(f"""_streaming_transcribe={output}""")
            yield output

    def transcribe_with_diarization_file(self,filepath: str):
        self._stop_event.set()
        if self._current_process and self._current_process.is_alive():
            self._current_process.join(timeout=0.5)
        self._stop_event.clear()
        
        audio = whisperx.load_audio(filepath, 16000)
        # print(f"audio: {audio}  filepath= {filepath}.")
        if audio is None:
            print(f"Error: Failed to load audio from {filepath}.")     
        
        result_queue = queue.Queue()
        self._current_process = threading.Thread(
            target=self.transcribe,
            args=((16000, audio), result_queue)
        )
        self._current_process.start()
        
        while not self._stop_event.is_set():
            try:
                yield result_queue.get(timeout=0.1)
            except queue.Empty:
                if not self._current_process.is_alive():
                    break
                    
        # return self.transcribe(
            # (16000, audio), None, "", False
        # )
        
        
