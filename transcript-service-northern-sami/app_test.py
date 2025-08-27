import os
import gradio as gr
from vector_search import HybridVectorSearch
from whisper_asr import WhisperAutomaticSpeechRecognizer
import time
import tempfile
import json
import soundfile as sf
import shutil
from utils.audio_transcript_utils import *
 
 
with gr.Blocks(css="""
    .custom-btn {
        width: 60px !important;    
        height: 40px !important;   
        font-size: 0.9rem !important;
        border-radius: 0.375rem !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .file-btn {  
        display: flex !important;  
        border-radius: 0.375rem !important;
    }
    .transcriptBox {
        position: relative;
    }
    .btn-container {
        position: absolute;
        bottom: 0.5rem;
        right: 0.5rem;
        display: flex;
        gap: 0.25rem;
    }
    .gr-textbox { overflow-y: auto; }
    
    /* 使用CSS选择器识别并高亮标记文本 */
    .gr-textbox textarea::selection {
        background-color: #e0f2fe;
    }
    
    /* 使用JavaScript将标记转换为实际高亮 */
    .highlight {
        background-color: #e0f2fe;
        transition: background-color 0.3s ease;
    }
    .transcript-container {
        overflow-y: auto;
        height: 100%;
        padding-right: 1rem;
    }
    .transcript-line {
        padding: 4px;
        cursor: pointer;
    }
    .transcript-line.highlight {
        background-color: #e0f2fe;
        transition: background-color 0.3s ease;
    }
    .edit-container {
        position: relative;
        margin-top: 1rem;
    }
    .edit-textarea {
        min-height: 100px;
    }
    .edit-controls {
        margin-top: 0.5rem;
        display: flex;
        justify-content: flex-end;
        gap: 0.5rem;
    }
"""
) as demo:
    full_stream = gr.State()
    transcript = gr.State(value="")
    chats = gr.State(value=[])
    edit_mode = gr.State(value=False)
    original_text = gr.State(value="")
    
    audio_timestamps = gr.State([])
    current_audio = gr.State(None)

    
    def toggle_edit_mode(edit_mode, current_text):
        new_mode = not edit_mode
        return (
            new_mode, 
            gr.Textbox(interactive=new_mode),  
            gr.Button(visible=not new_mode),  
            gr.Button(visible=new_mode),  
            gr.Button(visible=new_mode, variant="secondary", value="Cancel"),  
            current_text if new_mode else original_text.value,
            current_text  
        )

    def cancel_edit(edit_mode, original_text):
        return (
            False,  
            gr.Textbox(interactive=False, value=original_text),  
            gr.Button(visible=True), 
            gr.Button(visible=False), 
            gr.Button(visible=False),  
            original_text 
        )
    def save_edited_transcript(transcript_text, edit_mode, original_text):
        text_to_save = transcript_text if transcript_text or edit_mode else original_text
        saved_file = save_transcript(transcript_text)
        if not saved_file and edit_mode:
            saved_file = save_transcript(original_text)
  
        # insert_success = HybridVectorSearch.insert_text(
            # text=text_to_save,
            # metadata={"source": "live_transcription", "edited": edit_mode}
        # )
        # if not insert_success:
            # print("warning: save to qdrant fail")
  
        return (
            False,  
            gr.Textbox(interactive=False, value=transcript_text),
            gr.Button(visible=True),  # 显示Edit按钮
            gr.Button(visible=False),  # 隐藏Save按钮
            gr.Button(visible=False),  # 隐藏Cancel按钮
            transcript_text if edit_mode else original_text,  
            saved_file  
        ) 
        
    # def insert_transcript_to_vector_db(transcript_text):
        # if transcript_text:
            # HybridVectorSearch.insert_text(transcript_text)
        # return transcript_text
        
    with gr.Tab("Live Mode"):
        with gr.Row(variant="panel"):
            audio_input = gr.Audio(sources=["microphone"], streaming=True)
            with gr.Row():
                save_audio_btn = gr.Button("Save Audio", visible=True, elem_classes="custom-btn")
                saved_audio_file = gr.File(label="audio_file", visible=True, file_count="single", type="filepath", elem_classes="file-btn")

        with gr.Row(variant="panel", equal_height=True):
            # with gr.Column(scale=1):
                # chatbot = gr.Chatbot(
                    # bubble_full_width=True, height="65vh", show_copy_button=True
                # )
                # chat_input = gr.Textbox(
                    # interactive=True, placeholder="Type Search Query...."
                # )
            with gr.Column(scale=1):
                transcript_textbox = gr.Textbox(
                    lines=40,
                    placeholder="Transcript",
                    max_lines=40,
                    label="Transcript",
                    show_label=True,
                    autoscroll=True,
                )
                with gr.Row():
                    edit_button = gr.Button("Edit Transcript", elem_classes="custom-btn")
                    cancel_edit_btn = gr.Button("Cancel", visible=False, elem_classes="custom-btn", variant="secondary")
                    save_transcript_btn = gr.Button("Save Transcript", visible=True, elem_classes="custom-btn")
                    saved_transcript_file = gr.File(label="transcript_file", visible=True, file_count="single", type="filepath", elem_classes="file-btn")

        # chat_input.submit(
            # HybridVectorSearch.chat_search, [chat_input, chatbot], [chat_input, chatbot]
        # )
        audio_input.stream(
            WhisperAutomaticSpeechRecognizer.transcribe_with_diarization,
            [audio_input, full_stream, transcript],
            [transcript_textbox, full_stream, transcript],
        )
        save_audio_btn.click(
            save_audio,
            inputs=[audio_input],
            outputs=[saved_audio_file],
            show_progress=True,
            # 强制刷新
            api_name="save_audio"
        )
        edit_button.click(
            toggle_edit_mode,
            inputs=[edit_mode, transcript_textbox],
            outputs=[edit_mode, transcript_textbox, edit_button, save_transcript_btn, cancel_edit_btn, original_text, transcript_textbox],
            show_progress=False
        )
        cancel_edit_btn.click(
            cancel_edit,
            inputs=[edit_mode, original_text],
            outputs=[edit_mode, transcript_textbox, edit_button, save_transcript_btn, cancel_edit_btn, original_text],
            show_progress=False
        )
 
        save_transcript_btn.click(
            save_edited_transcript,
            inputs=[transcript_textbox, edit_mode, original_text],
            outputs=[edit_mode, transcript_textbox, edit_button, save_transcript_btn, cancel_edit_btn, original_text, saved_transcript_file],
            show_progress=True,
            api_name="save_transcript"
        )
        
        # transcript_textbox.change(
            # insert_transcript_to_vector_db,
            # inputs=[transcript_textbox],
            # outputs=[transcript_textbox]
        # )        
        
    with gr.Tab("Offline Mode"):
        with gr.Row(variant="panel"):
            audio_input = gr.Audio(
                sources=["upload"], 
                type="filepath"
            )
            audio_time = gr.Slider(minimum=0, maximum=1000, value=0, step=0.1, visible=False)
            
        with gr.Row(variant="panel", equal_height=True):
            # with gr.Column(scale=1):
                # chatbot = gr.Chatbot(
                    # bubble_full_width=True, height="55vh", show_copy_button=True
                # )
                # chat_input = gr.Textbox(
                    # interactive=True, placeholder="Type Search Query...."
                # )
            with gr.Column(scale=1):
                transcript_textbox = gr.Textbox(
                    lines=35,
                    placeholder="Transcripts",
                    max_lines=35,
                    label="Transcript",
                    show_label=True,
                    autoscroll=True,
                    # visible=False 
                )
                with gr.Row():
                    edit_button = gr.Button("Edit Transcript", elem_classes="custom-btn")
                    save_transcript_btn = gr.Button("Save Transcript", visible=True, elem_classes="custom-btn")
                    cancel_edit_btn = gr.Button("Cancel", visible=False, elem_classes="custom-btn", variant="secondary")
                    saved_transcript_file = gr.File(label="transcript_file", visible=True, file_count="single", type="filepath", elem_classes="file-btn")

        # chat_input.submit(
            # HybridVectorSearch.chat_search, [chat_input, chatbot], [chat_input, chatbot]
        # )
        def offline_audio_handler(audio_file):
            if not audio_file:
                return "", [], None, 0
            transcript_text, timestamps = process_audio_with_timestamps(audio_file)
            current_audio.value = audio_file
            print(f"offline_audio_handler={transcript_text}")
            return transcript_text, timestamps, audio_file, sf.info(audio_file).duration
        
        audio_input.upload(
            # WhisperAutomaticSpeechRecognizer.transcribe_with_diarization_file,
            offline_audio_handler,
            [audio_input],
            # [transcript_textbox, full_stream, transcript],
            [transcript_textbox, audio_timestamps, audio_input, audio_time]
        )

        def update_text_highlight(current_time, timestamps, transcript_text):
            if not timestamps or current_time is None:
                return transcript_text
            highlighted_text = update_highlight(current_time, timestamps, transcript_text)
            return highlighted_text
        
        audio_time.change(
            update_text_highlight,
            inputs=[audio_time, audio_timestamps, transcript_textbox],
            outputs=[transcript_textbox],
        )
        # 同步音频播放时间和滑块
        demo.load(
            None, None, None,
            js="""
            () => {
                const audioEl = document.querySelector('audio');
                const timeSlider = document.querySelector('input[type="range"]');
                const textbox = document.querySelector('.gr-textbox textarea');
                
                if (audioEl && timeSlider && textbox) {
                    // 音频播放时更新滑块
                    audioEl.addEventListener('timeupdate', () => {
                        timeSlider.value = audioEl.currentTime;
                        timeSlider.dispatchEvent(new Event('input'));
                    });
                    
                    // 滑块拖动时更新音频位置
                    timeSlider.addEventListener('input', () => {
                        if (audioEl) audioEl.currentTime = parseFloat(timeSlider.value);
                    });
                    
                    // 定期检查并更新高亮
                    setInterval(() => {
                        if (!audioEl.paused && audioEl.currentTime > 0) {
                            const text = textbox.value;
                            if (text && text.includes('[HIGHLIGHT]')) {
                                // 移除旧的高亮
                                const ranges = window.getSelection().getRangeAt(0);
                                const content = textbox.value;
                                
                                // 替换高亮标记为实际的高亮样式
                                const highlightedContent = content
                                    .replace(/\u200B\[HIGHLIGHT\]\u200B/g, '<span class="highlight">')
                                    .replace(/\u200B\[\/HIGHLIGHT\]\u200B/g, '</span>');
                                
                                // 使用execCommand应用样式
                                document.designMode = 'on';
                                textbox.focus();
                                document.execCommand('selectAll');
                                document.execCommand('insertHTML', false, highlightedContent);
                                document.designMode = 'off';
                                
                                // 恢复选择范围
                                window.getSelection().addRange(ranges);
                            }
                        }
                    }, 300);
                }
            }
            """
        )
        
        edit_button.click(
            toggle_edit_mode,
            inputs=[edit_mode, transcript_textbox],
            outputs=[edit_mode, transcript_textbox, edit_button, save_transcript_btn, cancel_edit_btn, original_text, transcript_textbox],
            show_progress=False
        )
      
        cancel_edit_btn.click(
            cancel_edit,
            inputs=[edit_mode, original_text],
            outputs=[edit_mode, transcript_textbox, edit_button, save_transcript_btn, cancel_edit_btn, original_text],
            show_progress=False
        )
        
        save_transcript_btn.click(
            save_edited_transcript,
            inputs=[transcript_textbox, edit_mode, original_text],
            outputs=[edit_mode, transcript_textbox, edit_button, save_transcript_btn, cancel_edit_btn, original_text, saved_transcript_file],
            show_progress=True,
            api_name="save_transcript"
        )
   
        
if __name__ == "__main__":
    # HybridVectorSearch.initialize()
    # if not HybridVectorSearch.verify_data_insertion():
        # HybridVectorSearch.check_and_add_sample_data()
        
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
