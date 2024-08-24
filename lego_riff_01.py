# use env => latest python

# took me about 2 days to produce the first 2 riffs(with the same key)


# NEXT:
# 1)create_midi_lego_riff_combi -> 
    # try to use notes_df(output) from create_lego_riff_note_combi
        # build on top of create_midi_lego_riff_1file that will create midi files with different scales, keys, bpm in a loop with ost

# 2)convert midi directly to mp3(not important bc I can use format factory for now)
# try to do that manually(Clade Music_with_Code_03)

# from music21 import stream
# import pandas as pd
# from midi2audio import FluidSynth
# import fluidsynth
# from typing import *
from pathlib import Path 
from music_func import *
# from mingus.core import scales, notes, intervals
# from functools import partial
# from mingus.containers import Note
# import numpy as np
# import inspect_py as inp
# import pandas as pd


# notes.int_to_note()
# scales._Scale
# based on migus 0.6.1

OUTPUT_FOLDER = Path(r"C:\Users\Heng2020\OneDrive\D_Code\Python\Python Music\2024\01 Lego Riff Creation\lego_riff_creation\test_output")


####################################

def create_block01():
    note_lengths01 = [1,0.5,0.5,1,2]
    note_names01 = ["D4", " E4", " D4", " C4", " D4"]
    OUTPUT_FOLDER = Path(r"C:\Users\Heng2020\OneDrive\D_Code\Python\Python Music\2024\01 Lego Riff Creation\lego_riff_creation\test_output")
    OUTPUT_PATH01 = OUTPUT_FOLDER/ 'test01.mid'
    create_midi(OUTPUT_PATH01,note_names01,note_lengths01)

def create_block02():
    note_lengths01 = [0.5,0.5,1]
    note_names01 = ["C5","Bb4","Ab4"]
    OUTPUT_PATH01 = OUTPUT_FOLDER/ 'riff_02_block.mid'
    create_midi(OUTPUT_PATH01,note_names01,note_lengths01)

def create_riff01():
    block01 = [2,3,2,1,2]
    note_lengths01 = [1,0.5,0.5,1,1]
    scale_degrees01 = make_num_seq(block01,7,as_np=False)
    riff_notes01 = convert_num_to_scale(scale_degrees01)

    OUTPUT_FOLDER = Path(r"C:\Users\Heng2020\OneDrive\D_Code\Python\Python Music\2024\01 Lego Riff Creation\lego_riff_creation\test_output")
    OUTPUT_PATH01 = OUTPUT_FOLDER/ 'test02_120bpm_long.mid'
    OUTPUT_PATH02 = OUTPUT_FOLDER/ 'test02_240bpm_long.mid'

    create_midi_repeate_tempo(OUTPUT_PATH01,riff_notes01,note_lengths01)
    create_midi_repeate_tempo(OUTPUT_PATH02,riff_notes01,note_lengths01,bpm=240)

def create_riff01_variation():
    block01 = [2,3,2,1,2]
    block01_reversed = block01[::-1]

    note_lengths01 = [1,0.5,0.5,1,1]
    note_lengths01_reversed = note_lengths01[::-1]

    scale_degrees01 = make_num_seq(block01,7,as_np=False)
    scale_degrees01_reversed = make_num_seq(block01_reversed,7,as_np=False)

    riff_notes01 = convert_num_to_scale(scale_degrees01)
    riff_notes01_reversed = convert_num_to_scale(scale_degrees01_reversed)


    OUTPUT_FOLDER = Path(r"C:\Users\Heng2020\OneDrive\D_Code\Python\Python Music\2024\01 Lego Riff Creation\lego_riff_creation\test_output\reverse_riff")
    OUTPUT_PATH01 = OUTPUT_FOLDER/ 'test01_normal.mid'
    OUTPUT_PATH02 = OUTPUT_FOLDER/ 'test01_reverse_tempo.mid'
    OUTPUT_PATH03 = OUTPUT_FOLDER/ 'test01_reverse_note.mid'
    OUTPUT_PATH04 = OUTPUT_FOLDER/ 'test01_reverse_all.mid'

    create_midi_repeate_tempo(OUTPUT_PATH01,riff_notes01,note_lengths01,bpm=240)
    create_midi_repeate_tempo(OUTPUT_PATH02,riff_notes01,note_lengths01_reversed,bpm=240)
    create_midi_repeate_tempo(OUTPUT_PATH03,riff_notes01_reversed,note_lengths01,bpm=240)
    create_midi_repeate_tempo(OUTPUT_PATH04,riff_notes01_reversed,note_lengths01_reversed,bpm=240)
    print()

def create_riff01_variation2():
    from itertools import permutations, combinations
    block01 = [2,3,2,1,2]
    block01_reversed = block01[::-1]

    note_lengths01 = [1,0.5,0.5,1,1]
    note_lengths01_reversed = note_lengths01[::-1]

    # rhythm_set = [
    #     (1,   0.5,  0.5, 1,   1),
    #     (1,   0.5,  1,   0.5, 1),
    #     (1,   1,    1,   0.5, 0.5),

    # ]

    rhythm_set = list(set(permutations(note_lengths01)))
    scale_degrees01 = make_num_seq(block01,7,as_np=False)
    scale_degrees01_reversed = make_num_seq(block01_reversed,7,as_np=False)

    riff_notes01 = convert_num_to_scale(scale_degrees01)
    riff_notes01_reversed = convert_num_to_scale(scale_degrees01_reversed)

    # this_file = __file__
    OUTPUT_FOLDER = Path(r"C:\Users\Heng2020\OneDrive\D_Code\Python\Python Music\2024\01 Lego Riff Creation\lego_riff_creation\test_output\multiple_rhythm")
    OUTPUT_PATH02 = OUTPUT_FOLDER/ 'test01_reverse_tempo.mid'
    OUTPUT_PATH03 = OUTPUT_FOLDER/ 'test01_reverse_note.mid'
    OUTPUT_PATH04 = OUTPUT_FOLDER/ 'test01_reverse_all.mid'

    for i, rhythm in enumerate(rhythm_set):
        OUTPUT_PATH = OUTPUT_FOLDER/ f'rhythm{i+1}.mid'
        create_midi_repeate_tempo(OUTPUT_PATH,riff_notes01,rhythm,bpm=240)

    # create_midi_repeate_tempo(OUTPUT_PATH02,riff_notes01,note_lengths01_reversed,bpm=240)
    # create_midi_repeate_tempo(OUTPUT_PATH03,riff_notes01_reversed,note_lengths01,bpm=240)
    # create_midi_repeate_tempo(OUTPUT_PATH04,riff_notes01_reversed,note_lengths01_reversed,bpm=240)
    print()


def create_riff02():
    block01 = [3,2,1]
    note_lengths01 = [0.75,0.25,1]
    scale_degrees01 = make_num_degree_down(block01,5)
    riff_notes01 = convert_num_to_scale(scale_degrees01,scale_type="Minor")

    OUTPUT_PATH01 = OUTPUT_FOLDER/ 'riff_02_200bpm.mid'
    OUTPUT_PATH02 = OUTPUT_FOLDER/ 'riff_02_240bpm.mid'

    create_midi_repeate_tempo(OUTPUT_PATH01,riff_notes01,note_lengths01,bpm=200)
    create_midi_repeate_tempo(OUTPUT_PATH02,riff_notes01,note_lengths01,bpm=240)

def create_riff03():
    block01 = [2,3,2,1,2]
    note_lengths01 = [1,0.5,0.5,1,1]
    scale_degrees01 = make_num_degree_down(block01,7,root_degree=1)
    riff_notes01 = convert_num_to_scale(scale_degrees01)

    OUTPUT_PATH01 = OUTPUT_FOLDER/ 'riff_03_200bpm.mid'
    OUTPUT_PATH02 = OUTPUT_FOLDER/ 'riff_03_240bpm.mid'

    create_midi_repeate_tempo(OUTPUT_PATH01,riff_notes01,note_lengths01,bpm=200)
    create_midi_repeate_tempo(OUTPUT_PATH02,riff_notes01,note_lengths01,bpm=240)


def main_real_test():
    create_riff01_variation2()
    # create_riff01_variation()
    # create_riff03()
    # create_riff02()
    # create_block02()
    # create_riff01()
    # create_block01()



def main():
    main_real_test()
    
if __name__ == '__main__':
    main()