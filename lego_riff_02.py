from pathlib import Path 
from music_func import *

OUTPUT_FOLDER = Path(r"H:\D_Music\Riffs & Runs\Python Riff")

def create_riff03():
    block01 = [1,3,4,5,6,5,4,3]
    note_lengths01 = [0.75,0.25,1]
    scale_degrees01 = make_num_degree_down(block01,5)
    riff_notes01 = convert_num_to_scale(scale_degrees01,key="C")

    OUTPUT_PATH01 = OUTPUT_FOLDER/ 'riff_03_key_C_200bpm.mid'
    OUTPUT_PATH02 = OUTPUT_FOLDER/ 'riff_03_key_C_240bpm.mid'

    create_midi_repeate_tempo(OUTPUT_PATH01,riff_notes01,note_lengths01,bpm=200)
    create_midi_repeate_tempo(OUTPUT_PATH02,riff_notes01,note_lengths01,bpm=240)

def main():
    create_riff03()
    
if __name__ == '__main__':
    main()