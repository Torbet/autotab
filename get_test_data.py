from guitarset_interpreter import tokenize_jams
import jams
from audio import load_audio, SAMPLE_RATE
from scrapers.get_model_data import get_vocab
from tokenizer import encoder
import os
import librosa
import numpy as np

example = [(14,'<S0><F3>'), (22,'<S0><F3>'), (28, '<S0><F3>'), (31, '<S0><F3>')]

audio_dir = 'data/guitarset-data/audio_mono-mic'
annot_dir = 'data/guitarset-data/annotation'

if __name__ == '__main__':
    max_token_seg_len = 1000

    solo_audio, solo_tabs = [], []
    comp_audio, comp_tabs = [], []

    enc = encoder()
    vocab = get_vocab(enc)
    all_unknown_tokens = set()

    solo_count, comp_count = 0, 0
    
    for filename in os.listdir(audio_dir):
        solo = 'solo' in filename

        audio_file = os.path.join(audio_dir, filename)
        annot_file = os.path.join(annot_dir, filename.replace('_mic.wav', '.jams'))
        # check annot file exists
        if not os.path.exists(annot_file):
            print(annot_file)
            continue

        waveform, _ = librosa.load(audio_file, sr=SAMPLE_RATE)
        spec = load_audio(waveform)

        jam = jams.load(annot_file)
        tokens = [t[1] for t in tokenize_jams(jam) if t[0] < 30]

        tokens_set = set(tokens)
        unknown_tokens = tokens_set - vocab

        if unknown_tokens:
            all_unknown_tokens.update(unknown_tokens)
            print(f"Found {len(unknown_tokens)} unknown tokens in {filename}:")
            continue

        encoded_tokens = []
        tokens = ['<|startoftab|>'] + tokens + ['<|endoftab|>']
        for t in tokens:
            encoded_tokens += enc.encode(t, allowed_special=enc.special_tokens_set)

        if len(encoded_tokens) > max_token_seg_len:
            print(f"Skipping {filename} due to token length {len(encoded_tokens)}")
            continue

        encoded_tokens = np.pad(encoded_tokens, (0, max_token_seg_len - len(encoded_tokens)), 'constant')

        if solo:
            solo_count += 1
            solo_audio.append(spec)
            solo_tabs.append(encoded_tokens)
        else:
            comp_count += 1
            comp_audio.append(spec)
            comp_tabs.append(encoded_tokens)
        
        print(f"Processed {filename}")

    np.savez_compressed('data/model_data/test_data_solo.npz', tabs=np.array(solo_tabs, dtype=np.uint16), audio=np.array(solo_audio, dtype=np.float32))
    np.savez_compressed('data/model_data/test_data_comp.npz', tabs=np.array(comp_tabs, dtype=np.uint16), audio=np.array(comp_audio, dtype=np.float32))

    print(f"Processed {solo_count} solo files and {comp_count} comp files")
    print(f"Found {len(all_unknown_tokens)} unknown tokens:")
    for token in sorted(all_unknown_tokens):
        print(f"  - '{token}'")
