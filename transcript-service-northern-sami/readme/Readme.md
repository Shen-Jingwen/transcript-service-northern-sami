
## add wen 20250827
## more details please check Speech_transcription_service_for_northern_sami（Report）.pdf

Implements a Northern Sami speech transcription service. Frontend  built with Gradio Blocks, provides audio upload, model selection, transcript display, editing, and saving. It supports offline transcription with word-level timestamp alignment and audio-text highlighting. Users can switch between different ASR models, and long audio is processed in chunks for segmented output.



##activate environment
conda activate live_speech_0

##start/restart the service
./start.sh --daemon

##Logging and Debugging
tail -f log.txt

##wait a little bit and open log.txt to get the public link of the service









# References
- https://github.com/SubtleParesh/live-speech-reference-search
- https://github.com/MahmoudAshraf97/ctc-forced-aligner/tree/main
- https://github.com/NVIDIA/NeMo/tree/main/tools/nemo_forced_aligner
- https://huggingface.co/GetmanY1/wav2vec2-base-sami-cont-pt-22k-finetuned
- https://huggingface.co/GetmanY1/wav2vec2-large-sami-cont-pt-22k-finetuned
