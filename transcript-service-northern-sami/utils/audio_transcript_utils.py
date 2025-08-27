import json
import time
import tempfile
import soundfile as sf
import shutil
from bs4 import BeautifulSoup
# from whisper_asr import WhisperAutomaticSpeechRecognizer
from asr_instance import asr
from .forced_alignment import generate_word_timestamps



def save_transcript(transcript_text):
    if not transcript_text or not isinstance(transcript_text, str):
        print(f"Error: Invalid transcript text, type()={type(transcript_text)} transcript_text={transcript_text}")
        return None
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", prefix=f"transcript_{timestamp}_")
    temp_file.close()
    # print(transcript_text)
    # print(temp_file.name)
    # print(temp_file)
    with open(temp_file.name, 'w', encoding='utf-8') as f:
        f.write(transcript_text)
    return temp_file.name


def save_audio(audio_data):
    if audio_data is None:
        print("Error: No audio data provided")
        return None
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", prefix=f"recording_{timestamp}_")
    temp_file.close()
    print(f"Saving audio to: {temp_file.name}")

    if isinstance(audio_data, tuple):
        sr, y = audio_data
        sf.write(temp_file.name, y, sr)
    else:
        shutil.copy(audio_data, temp_file.name)

    return temp_file.name



def resolve_text_to_save(edited_text, plain_text):
    return save_transcript(edited_text if edited_text.strip() else plain_text)


def time_str_to_seconds(time_str):
    """
    time transfer（0:00、00:00:00）to seconds
    """
    # print(f"time_str_to_seconds={time_str}")
    parts = list(map(int, time_str.split(':')))
    if len(parts) == 2:  # 0:11
        minutes, seconds = parts
        return minutes * 60 + seconds 
    elif len(parts) == 3:  # 1:10:03
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds*1.0 
    else:
        raise ValueError(f"time error: {time_str}")

   
def seconds_to_time_str(s):
    """
    time transfer seconds to (0:00、00:00:00)
    """
    if s is None or s == 0:
        return "0:00"
    try:
        s = float(s)
    except (ValueError, TypeError):
        return "0:00"
    hours = int(s / 3600)
    minutes = int((s % 3600) / 60)
    seconds = int(s % 60)
    if hours > 0:
        # HH:MM:SS
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        # MM:SS
        return f"{minutes:02d}:{seconds:02d}"    



def get_word_timestamp(word, previousWord, nextWord, timestamps):
    # print (f"""double click {word}  pre/next {previousWord} {nextWord}  timestamps={timestamps} """)
    if not word or not timestamps:
        return 0.0, "0:00"
    timestamps = list(timestamps)
    for index, item in enumerate(timestamps):
        if item['text'].strip() == word.strip():
            print (f"""index {index}  timestamps[index]={timestamps[index]}""")
            if index == 0:
                if timestamps[index+1]['text'].strip() == nextWord.strip():
                    word_timestamp = float(item['start'])
                    str_time=seconds_to_time_str(word_timestamp)
                    print (f"""double click {word}  word_timestamp={word_timestamp}={str_time} pre/next {previousWord} {nextWord}""")
                    # return float(item['start'])
                    return word_timestamp, str_time
            elif nextWord in [ None, "None"]:
                if timestamps[index-1]['text'].strip() == previousWord.strip():
                    word_timestamp = float(item['start'])
                    str_time=seconds_to_time_str(word_timestamp)
                    print (f"""double click {word}  word_timestamp={word_timestamp}={str_time} pre/next {previousWord} {nextWord}""")
                    # return float(item['start'])
                    return word_timestamp, str_time
            else:
                if timestamps[index-1]['text'].strip() == previousWord.strip() and timestamps[index+1]['text'].strip() == nextWord.strip():
                    word_timestamp = float(item['start'])
                    str_time=seconds_to_time_str(word_timestamp)
                    print (f"""double click {word}  word_timestamp={word_timestamp}={str_time} pre/next {previousWord} {nextWord}""")
                    # return float(item['start'])
                    return word_timestamp, str_time
    return 0.0, "0:00"


'''
def process_audio_with_timestamps(audio_file):
    if not audio_file:
        return "", []

    result = asr.transcribe_with_diarization_file(filepath=audio_file)
    # print(f"process_audio_with_timestamps={result}")

    try:
        if isinstance(result, tuple):
            transcript_text = result[0]
        else:
            transcript_text = result
        timestamps = generate_word_timestamps(audio_file, transcript_text)
        print(f"Extracted timestamps: {timestamps}")
    except Exception as e:
        print(f"Error processing timestamps: {e}")
        duration = sf.info(audio_file).duration if audio_file else 1000.0
        timestamps = [(0.0, duration, transcript_text)]

    return transcript_text, timestamps
'''
def process_audio_with_timestamps(audio_file):
    if not audio_file:
        yield "", []
        return
    asr._stop_event.set()
    if asr._current_process and asr._current_process.is_alive():
        asr._current_process.join(timeout=1.0)
    
    transcript_text = ""
    timestamps = []
    last_result = None
    
    asr._stop_event.clear()
    try:  
        for result in asr.transcribe_with_diarization_file(audio_file):
            if asr._stop_event.is_set():  
                break
            transcript_text = result[0]
            timestamps = generate_word_timestamps(audio_file, transcript_text)
            last_result = (transcript_text, timestamps)
            yield transcript_text, timestamps
        if not asr._stop_event.is_set() and last_result:
            yield last_result
    except Exception as e:
        duration = sf.info(audio_file).duration if audio_file else 1000.0
        yield transcript_text, [(0.0, duration, transcript_text)] if transcript_text else []


def update_highlight(current_time, timestamps, original_text):
    """  """
    if not timestamps or current_time is None:
        print(f'update_highlight time:timestamps{timestamps} current_time={current_time}')
        return original_text
    try:
        current_time = float(current_time)
    except (TypeError, ValueError):
        print(f'Invalid current_time: {current_time}')
        return original_text
        
    highlighted_html = ""
    current_pos = 0

    for item in timestamps:
        start = float(item['start'])
        end = float(item['end'])
        word = item['text']
        # print(f"update_highlight={start, end, word}")
        if not word:
            continue

        # check word position in transcript
        word_pos = original_text.find(word, current_pos)
        if word_pos == -1:
            # cannot find word, add all
            highlighted_html += word
            current_pos += len(word)
            continue

        if word_pos > current_pos:
            highlighted_html += original_text[current_pos:word_pos]

        # check highlight
        if start <= current_time + 2.0:  #2s delay, need better solution
            highlighted_html += f'<span class="highlight">{word}</span>'
        else:
            highlighted_html += f'<span class="word">{word}</span>'

        current_pos = word_pos + len(word)

    # add left transcript
    if current_pos < len(original_text):
        highlighted_html += original_text[current_pos:]

    return f"<span>{highlighted_html}</span>"


# current_time = 1.0
# timestamps = [{'start':0.42, 'end':0.861, 'text':'ceahkki'},{'start':0.92, 'end':1.061, 'text':'lei'}]
# original_text = "<span>ceahkki lei<span>"
# highlighted_text = update_highlight(current_time, timestamps, original_text)
# print(highlighted_text)


