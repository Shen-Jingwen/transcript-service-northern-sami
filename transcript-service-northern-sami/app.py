import os
import gradio as gr
from vector_search import HybridVectorSearch
# from whisper_asr import WhisperAutomaticSpeechRecognizer
from custom_asr import CustomAutomaticSpeechRecognizer, ASR_MODELS
from asr_instance import asr, default_asr
import time
import tempfile
import json, re
import soundfile as sf
import shutil
from utils.audio_transcript_utils import *
 

CSS="""
    .custom-btn {
        # width: 60px !important;    
        # height: 40px !important; 
        display: flex !important;          
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
    .transcript-stream {
        animation: fadeIn 0.5s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0.5; }
        to { opacity: 1; }
    }
    
    .highlight {
        background-color: #e0f2fe !important;
        border-radius: 3px !important;
        padding: 1px 3px;
        transition: background-color 0.2s ease;
    }
    .transcript-container {
        overflow-y: auto;
        height: 100%;
        padding: 10px;
        # padding-right: 1rem;
    }
    .transcript-line {
        # padding: 4px;
        # cursor: pointer;
        min-height: 1.2em;
        line-height: 1.2;
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

def retranscribe_on_model_change(model_name, audio_input_value, mode):
    asr._stop_event.set()
    if asr._current_process and asr._current_process.is_alive():
        asr._current_process.join(timeout=0.5)
    asr.change_model(model_name)
    asr._stop_event.clear()
    if mode == "live":
        if audio_input_value is not None:
            sr, audio = audio_input_value
            audio = asr.preprocess_audio(audio, sr)
            output = asr.pipe(audio, chunk_length_s=30, stride_length_s=5)['text']
            return output, audio_input_value, output, model_name
        return "", None, "", model_name
    elif mode == "offline":
        if audio_input_value is not None:
            for transcript_text, timestamps in process_audio_with_timestamps(audio_input_value):
                if asr._stop_event.is_set():
                    break
                duration = sf.info(audio_input_value).duration
                html_text = f'<span>{transcript_text}</span>'
                yield html_text, timestamps, duration, model_name, save_original_transcript(html_text)
            if not asr._stop_event.is_set():
                yield html_text, timestamps, duration, model_name, save_original_transcript(html_text)
        return "", [], 0, model_name, None

def toggle_edit_mode(edit_mode, current_text):
    new_mode = not edit_mode
    plain_text = remove_tags(current_text) if isinstance(current_text, str) else current_text
    
    return (
        new_mode, 
        gr.HTML(visible=not new_mode, value=current_text),
        gr.Textbox(visible=new_mode, interactive=True, lines=40, value=plain_text),  
        gr.Button(visible=not new_mode),  
        gr.Button(visible=new_mode),  
        gr.Button(visible=new_mode, variant="secondary", value="Cancel"),  
        current_text
    )

def edit_done(current_time, timestamps, transcript_text, edit_mode):
    text_to_save = remove_tags(transcript_text) if isinstance(transcript_text, str) else str(transcript_text)
    html_text = f'<span>{text_to_save}</span>'
    
    print(f"edit_done= text_to_save {text_to_save}")
    print(f"edit_done= html_text {html_text}")
    print(f"edit_done= gr.HTML {gr.HTML(value=html_text, visible=True)}")
    
    return (
        False,
        gr.HTML(value=html_text, visible=True),
        gr.Textbox(visible=False, value=text_to_save),
        gr.Button(visible=True),   # show Edit
        gr.Button(visible=False),  # hide Save
        gr.Button(visible=False),  # hide Cancel
        text_to_save
    )



def cancel_edit(edit_mode, original_text):
    plain_text = remove_tags(original_text) if isinstance(original_text, str) else original_text
    return (
        False,  
        gr.HTML(value=original_text, visible=True),  
        # gr.Textbox(interactive=False, value=original_text),  
        gr.Textbox(visible=False, value=plain_text),  
        gr.Button(visible=True), 
        gr.Button(visible=False), 
        gr.Button(visible=False),  
        plain_text 
    )
 
def remove_tags(text):
    tag_pattern = re.compile(r'<[^>]+>')
    return tag_pattern.sub('', text)
 
def save_edited_transcript(transcript_text, edit_mode, original_text):
    if edit_mode:
        text_to_save = transcript_text
        html_text = f'<span>{transcript_text}</span>' 
    else:
        text_to_save = remove_tags(transcript_text) if isinstance(transcript_text, str) else transcript_text
        html_text = transcript_text
    print(f"original_text={original_text}")
    print(f"transcript_text={transcript_text}")
    print(f"text_to_save={text_to_save}")
    print(f"html_text={html_text}")
    
    saved_file = save_transcript(text_to_save)
    if not saved_file and edit_mode:
        saved_file = save_transcript(original_text)
    if not edit_mode:
        return (
            not edit_mode,  
            # gr.Textbox(interactive=False, value=transcript_text),
            gr.HTML(value=html_text, visible=True),  
            gr.Textbox(visible=False, value=text_to_save),  
            gr.Button(visible=True),  # show Edit button
            gr.Button(visible=False),  # hide Save button
            gr.Button(visible=False),  # hide Cancel button
            text_to_save,  
            saved_file  
        ) 
    else:
        return (
            edit_mode,  
            # gr.Textbox(interactive=False, value=transcript_text),
            gr.HTML(value=html_text, visible=False),  
            gr.Textbox(visible=True, value=text_to_save),  
            gr.Button(visible=True),  # show Edit button
            gr.Button(visible=False),  # hide Save button
            gr.Button(visible=False),  # hide Cancel button
            text_to_save,  
            saved_file  
        ) 

def save_original_transcript(original_text):
    text_to_save = remove_tags(original_text) if isinstance(original_text, str) else original_text
    html_text = original_text
    saved_file = save_transcript(text_to_save)
    return saved_file  

def offline_audio_handler(audio_file):
    if not audio_file: 
        return "", "", "", [], None, 0
    asr._stop_event.set()
    current_audio.value = audio_file
    duration = sf.info(audio_file).duration if audio_file else 0
    asr._stop_event.clear()
    for transcript_text, timestamps in process_audio_with_timestamps(audio_file):
        if asr._stop_event.is_set():  # stop once recieve new audios
            break
        words = transcript_text.split()
        html_words = []
        for word in words:
            html_words.append(f'<span class="word">{word}</span>') #for dblclick event
        html_text = ' '.join(html_words)
        yield html_text, html_text, html_text, timestamps, audio_file, duration, save_original_transcript(html_text)
    
    if not asr._stop_event.is_set():
        yield html_text, html_text, html_text, timestamps, audio_file, duration, save_original_transcript(html_text)
    

def update_text_highlight(current_time, timestamps, transcript_text, transcript_html, edit_mode):
    # print(f"audio_time value: {type(current_time)} {current_time}")
    if isinstance(current_time, str): 
        if ":" in current_time:
            current_time=time_str_to_seconds(current_time)
            print(f"audio_time value: {type(current_time)} {current_time}")
    if not timestamps or current_time is None:
        # return transcript_text, transcript_text
        return transcript_text
  
    highlighted_text = update_highlight(current_time, timestamps, remove_tags(transcript_text) if isinstance(transcript_text, str) else transcript_text)
    # print(f"update_text_highlight={highlighted_text}")
    # return highlighted_text, remove_tags(transcript_text)
    return highlighted_text
        
# transcript box and edit button
def create_transcript_ui():
    with gr.Column(scale=25):
        # html for display
        transcript_html = gr.HTML(
            value="Transcript will appear here",
            label="Transcript",
            show_label=True,
            elem_id="transcript_html"
        )
        # Textbox for edit mode 
        transcript_edit = gr.Textbox(
            lines=40,
            placeholder="Transcript",
            max_lines=40,
            label="Transcript (Edit Mode)",
            show_label=True,
            visible=False,
            interactive=True,
            elem_id="transcript_textbox"
        )
    with gr.Column(scale=1):
        edit_button = gr.Button("Edit Transcript", elem_classes="custom-btn", elem_id="edit")
        edit_done_button = gr.Button("Save/Exit Edit", elem_classes="custom-btn", elem_id="edit_done")
        cancel_edit_btn = gr.Button("Cancel", visible=False, elem_classes="custom-btn", variant="secondary")
        original_transcript_file = gr.File(label="original_transcript", visible=True, file_count="single", type="filepath", elem_classes="file-btn")
        save_transcript_btn = gr.Button("Save Current Transcript", visible=True, elem_classes="custom-btn")
        saved_transcript_file = gr.File(label="new_saved_transcript", visible=True, file_count="single", type="filepath", elem_classes="file-btn")
    return transcript_html, transcript_edit, edit_button, edit_done_button, cancel_edit_btn, save_transcript_btn, saved_transcript_file, original_transcript_file

# edit button event
def bind_edit_buttons(edit_button, edit_done_button, cancel_edit_btn, save_transcript_btn, edit_mode, transcript_html, transcript_edit, original_text, current_time=None, timestamps=None):
    edit_button.click(
        toggle_edit_mode,
        inputs=[edit_mode, transcript_html],
        outputs=[edit_mode, transcript_html, transcript_edit, edit_button, save_transcript_btn, cancel_edit_btn, original_text],
        show_progress=False
    )
    edit_done_button.click(
        edit_done,
        inputs=[current_time, timestamps, transcript_edit, edit_mode],
        outputs=[edit_mode, transcript_html, transcript_edit, edit_button, save_transcript_btn, cancel_edit_btn, original_text],
        show_progress=False 
    )
    cancel_edit_btn.click(
        cancel_edit,
        inputs=[edit_mode, original_text],
        outputs=[edit_mode, transcript_html, transcript_edit, edit_button, save_transcript_btn, cancel_edit_btn, original_text],
        show_progress=False
    )
    save_transcript_btn.click(
        save_edited_transcript,
        inputs=[transcript_edit, edit_mode, original_text],
        outputs=[edit_mode, transcript_html, transcript_edit, edit_button, save_transcript_btn, cancel_edit_btn, original_text, saved_transcript_file],
        show_progress=True,
        api_name="save_transcript",
    )
  

with gr.Blocks(css=CSS) as demo:
    full_stream = gr.State()
    transcript = gr.State(value="")
    chats = gr.State(value=[])
    edit_mode = gr.State(value=False)
    original_text = gr.State(value="")
    
    audio_timestamps = gr.State([])
    current_audio = gr.State(None)

    model_selector = gr.Dropdown(
        choices=ASR_MODELS,
        value=default_asr,
        label="Select ASR Model"
    ) 
    '''
    with gr.Tab("Live Mode"):
        with gr.Row(variant="panel", equal_height=True):
            with gr.Column(scale=25):
                audio_input = gr.Audio(sources=["microphone"], streaming=True)
            with gr.Column(scale=1):
                save_audio_btn = gr.Button("Save Audio", visible=True, elem_classes="custom-btn")
                saved_audio_file = gr.File(label="audio_file", visible=True, file_count="single", type="filepath", elem_classes="file-btn")

        with gr.Row(variant="panel", equal_height=True):
            # transcript_textbox, edit_button, cancel_edit_btn, save_transcript_btn, saved_transcript_file = create_transcript_ui()
            transcript_html, transcript_edit, edit_button, edit_done_button, cancel_edit_btn, save_transcript_btn, saved_transcript_file, original_transcript_file = create_transcript_ui()

        audio_input.stream(
            # WhisperAutomaticSpeechRecognizer.transcribe_with_diarization,
            asr.transcribe,
            [audio_input, full_stream, transcript],
            [transcript_html, full_stream, transcript],
        )
        save_audio_btn.click(
            save_audio,
            inputs=[audio_input],
            outputs=[saved_audio_file],
            show_progress=True,
            api_name="save_audio"
        )
        bind_edit_buttons(edit_button, edit_done_button, cancel_edit_btn, save_transcript_btn, edit_mode, transcript_html, transcript_edit, original_text)
               
        model_selector.change(
            retranscribe_on_model_change,
            inputs=[model_selector, audio_input, gr.State("live")],
            outputs=[transcript_html, full_stream, transcript, model_selector]
        )
    '''    
    with gr.Tab("Offline Mode"):
        with gr.Row(variant="panel", equal_height=True):
            audio_input = gr.Audio(
                sources=["upload"], 
                type="filepath",
                interactive=True,
                elem_id="offline_audio"
            )
            audio_duration = gr.Number(value=0.0, visible=False, elem_id="audio_duration")
            played_duration = gr.Number(value=5.0, visible=False, elem_id="played_duration")
            audio_timestamps = gr.State([])
            
        with gr.Row(variant="panel", equal_height=True):
            transcript_html, transcript_edit, edit_button, edit_done_button, cancel_edit_btn, save_transcript_btn, saved_transcript_file, original_transcript_file = create_transcript_ui()

        audio_input.change(
            offline_audio_handler,
            [audio_input],
            [transcript_html, transcript_edit, original_text, audio_timestamps, current_audio, audio_duration, original_transcript_file],
            show_progress=True,  
        )

        bind_edit_buttons(edit_button, edit_done_button, cancel_edit_btn, save_transcript_btn, edit_mode, transcript_html, transcript_edit, original_text, played_duration, audio_timestamps)
        
        ### start alignment and highlight update ###
        hidden_trigger = gr.Button(visible=False, elem_id="hidden_trigger")
        hidden_trigger.click(
            time_str_to_seconds,  
            inputs=[played_duration],
            outputs=[played_duration],
            js="""() => {
                return [document.getElementById('time').textContent];
            }"""
        )      
        hidden_trigger.click(
            update_text_highlight,
            inputs=[played_duration, audio_timestamps, transcript_edit, transcript_html, edit_mode],
            # outputs=[transcript_html, transcript_edit],
            outputs=[],
            js="""
                function(played_duration, audio_timestamps, transcript_edit, transcript_html, edit_mode) {
                    const transcriptElement = document.getElementById('transcript_html');
                    const transcriptElement_edit = document.getElementById('transcript_textbox');
                    if (transcriptElement) {
                        transcriptElement.innerHTML = transcript_html;
                    }
                    //if (transcriptElement_edit) {
                        //transcriptElement_edit.innerHTML = transcript_edit;
                    //}
                    return [];
                }
                """    
        ) 
        audio_input.play(
            time_str_to_seconds,  
            inputs=[played_duration],
            outputs=[played_duration],
            js="""
                function() {
                    const intervalId = setInterval(() => {
                        const timeElement = document.getElementById('time');
                        if (timeElement) {
                            const played_duration = timeElement.textContent;
                            console.log('time:', played_duration);
                            
                            document.getElementById("hidden_trigger").click();
                            return [played_duration];  
                        }
                        return ["0:00"];  // default
                    }, 1000);  

                    document.getElementById('offline_audio').dataset.intervalId = intervalId;

                    // initial
                    const timeElement = document.getElementById('time');
                    return timeElement ? [timeElement.textContent] : ["0:00"];
                }
            """    
        )
        
        audio_input.play(
            update_text_highlight,
            inputs=[played_duration, audio_timestamps, transcript_edit, transcript_html, edit_mode],
            show_progress=True,
            # outputs=[transcript_html, transcript_edit],
            outputs=[],
            js="""
                function(played_duration, audio_timestamps, transcript_edit, transcript_html, edit_mode) {
                    const transcriptElement = document.getElementById('transcript_html');
                    const transcriptElement_edit = document.getElementById('transcript_textbox');
                    if (transcriptElement) {
                        transcriptElement.innerHTML = transcript_html;
                    }
                    //if (transcriptElement_edit) {
                        //transcriptElement_edit.innerHTML = transcript_edit;
                    //}
                    return [];
                }
            """    
        ) 

        played_duration.change(
            update_text_highlight,
            inputs=[played_duration, audio_timestamps, transcript_edit, transcript_html, edit_mode],
            # outputs=[transcript_html, transcript_edit]
            outputs=[transcript_html]
        )
        audio_input.pause(
            None,
            inputs=None,
            outputs=None,
            js="""
                function() {
                    const audioElement = document.getElementById('offline_audio');
                    if (audioElement.dataset.intervalId) {
                        clearInterval(parseInt(audioElement.dataset.intervalId));
                    }
                    return [];
                }
            """
        )
        edit_done_button.click(
            edit_done,
            inputs=[played_duration, audio_timestamps, transcript_edit, edit_mode],
            outputs=[edit_mode, transcript_html, transcript_edit, edit_button, save_transcript_btn, cancel_edit_btn, original_text],
            show_progress=False 
        )
        ### end alignment and highlight update ###

        
        model_selector.change(
            retranscribe_on_model_change,
            inputs=[model_selector, audio_input, gr.State("offline")],
            outputs=[transcript_html, audio_timestamps, audio_duration, model_selector, original_transcript_file],
            show_progress=True
        )

        ### start double click transcript words then jump to audio position and play ###
        word_click_trigger = gr.Textbox(visible=False, elem_id="word_click_trigger")
        word_timestamp = gr.Number(value=0.0, visible=False, elem_id="word_timestamp")
        str_time=gr.Textbox(visible=False, elem_id="str_time")
        previousWord = gr.Textbox(visible=False, elem_id="previousWord")
        nextWord = gr.Textbox(visible=False, elem_id="nextWord")
        
        
        word_hidden_btn = gr.Button(visible=False, elem_id="word_hidden_btn")
        word_hidden_btn.click(
            None,  
            inputs=None,
            outputs=[word_click_trigger, previousWord, nextWord],
            js="""() => {
                return [
                    document.getElementById('word_click_trigger').textContent,
                    document.getElementById('previousWord').textContent,
                    document.getElementById('nextWord').textContent,
                ];
            }"""
        )      
        
        t_hidden_btn = gr.Button(visible=False, elem_id="t_hidden_btn")
        t_hidden_btn.click(
            None,  
            inputs=[word_timestamp, str_time],
            outputs=[word_timestamp, str_time]
        )   
 
        audio_input.change(
            None,  
            inputs=None,
            outputs=[word_click_trigger, previousWord, nextWord],
            js="""
                function() {
                    function handleWordDoubleClick(e) {
                        const clickedWord = e.target;
                        const word = clickedWord.innerText.trim();
                        const previousWord = clickedWord.previousElementSibling?.innerText.trim() || 'None';
                        const nextWord = clickedWord.nextElementSibling?.innerText.trim() || 'None';
                        
                        const triggerElem = document.getElementById('word_click_trigger');
                        triggerElem.textContent=word;
                        document.getElementById("word_hidden_btn").click();
                        
                        // console.log('e:', previousWord);
                        console.log('double clicked:', word);
                        // console.log('e:', nextWord);
                        
                        document.getElementById('previousWord').textContent=previousWord;
                        document.getElementById('nextWord').textContent=nextWord;
                    }
                    document.getElementById('transcript_html').addEventListener('dblclick', function(e){handleWordDoubleClick(e);});
                    document.getElementById('transcript_textbox').addEventListener('dblclick', function(e){handleWordDoubleClick(e);});
                    
                    word_click_trigger=document.getElementById('word_click_trigger').textContent;
                    previousWord=document.getElementById('previousWord').textContent;
                    nextWord=document.getElementById('nextWord').textContent;
                    return [word_click_trigger, previousWord, nextWord];
                }
            """    
        )
        word_click_trigger.change(
            get_word_timestamp,
            inputs=[word_click_trigger, previousWord, nextWord, audio_timestamps],
            outputs=[word_timestamp, str_time]
        )
        word_click_trigger.change(
            None,
            inputs=None,
            outputs=None,
            js="""
                function() {
                    document.getElementById("t_hidden_btn").click();
                    return [];
                }
            """    
        )
        str_time.change(
            None,
            inputs=[word_timestamp, str_time, audio_duration],
            outputs=[],
            js="""
                function(word_timestamp, str_time, audio_duration) {
                    if (word_timestamp > 0.0) {
                        const audioElement = document.getElementById('offline_audio');
                        document.getElementById('time').textContent = str_time;
                        const percentage = (word_timestamp / audio_duration) ;
                        console.log('Jumped to str_time:', str_time, audio_duration, percentage);
                        
                        const gradioApp = document.querySelector('gradio-app');
                        const root = gradioApp.shadowRoot || gradioApp;
                        const audioContainer = root.querySelector('#offline_audio');
                        
                        const waveform = document.getElementById('waveform');
                        if (waveform) {
                            const rect = waveform.getBoundingClientRect();
                            if (audio_duration > 0) {
                                const clickX = rect.left + (rect.width * percentage);
                                const clickY = rect.top + (rect.height / 2);
                                
                                const clickEvent = new MouseEvent('click', {
                                    bubbles: true,
                                    cancelable: true,
                                    clientX: clickX,
                                    clientY: clickY,
                                    view: window
                                });
                                
                                waveform.dispatchEvent(clickEvent);
                                console.log('Dispatched click event on waveform');
                            }
                        }
                        
                        //if (audioContainer) {
                            //const playButton = audioContainer.querySelector('button[aria-label="Play"]');
                            //if (playButton) {
                               //playButton.click();
                            //}
                        //}
                    }
                    return [];
                }
            """
        )
        ### end double click transcript words then jump to audio position and play ###
                   
if __name__ == "__main__": 
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=True,
        ssl_verify=False
    )
