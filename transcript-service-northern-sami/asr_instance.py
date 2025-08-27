from custom_asr import CustomAutomaticSpeechRecognizer, ASR_MODELS
import torch
torch.cuda.empty_cache()

default_asr=ASR_MODELS[1]
asr = CustomAutomaticSpeechRecognizer(default_asr)