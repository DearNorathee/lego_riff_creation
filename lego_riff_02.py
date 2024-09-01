from pathlib import Path 
from music_func import *

OUTPUT_FOLDER = Path(r"H:\D_Music\Riffs & Runs\Python Riff")

def create_riff03():
    block01 = [1,3,4,5,6,5,4,3]
    note_lengths01 = [1,1,1,1,1,1,1,1]
    scale_degrees01 = make_num_seq(block01,6)
    # scale_degrees01 = make_num_degree_down(block01,5)
    riff_notes01 = convert_num_to_scale(scale_degrees01,key="C")

    OUTPUT_PATH01 = OUTPUT_FOLDER/ 'riff_03_key_C_300bpm.mid'
    OUTPUT_PATH02 = OUTPUT_FOLDER/ 'riff_03_key_C_350bpm.mid'

    create_midi_repeate_tempo(OUTPUT_PATH01,riff_notes01,note_lengths01,bpm=300)
    create_midi_repeate_tempo(OUTPUT_PATH02,riff_notes01,note_lengths01,bpm=350)

def create_riff04():
    block01 = [1,3,5,6,4,5,3,4]
    note_lengths01 = [1,1,1,1,1,1,1,1]

    block02 = [5,3,1,-1,2,1,3,2]
    note_lengths02 = [1,1,1,1,1,1,1,1]

    scale_degrees01 = make_num_seq(block01,6)
    scale_degrees02 = make_num_degree_down(block02,5,root_degree=1)

    riff_notes01_up = convert_num_to_scale(scale_degrees01,key="C")
    riff_notes02_down = convert_num_to_scale(scale_degrees02,key="C")

    OUTPUT_PATH01 = OUTPUT_FOLDER/ 'riff_04_key_C_up_350bpm.mid'
    OUTPUT_PATH02 = OUTPUT_FOLDER/ 'riff_04_key_C__down_reverse_350bpm.mid'

    create_midi_repeate_tempo(OUTPUT_PATH01,riff_notes01_up,note_lengths01,bpm=350)
    create_midi_repeate_tempo(OUTPUT_PATH02,riff_notes02_down,note_lengths01,bpm=350)

def main():
    create_riff04()
    # create_riff03()
    
if __name__ == '__main__':
    main()