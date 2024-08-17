# use env => latest python

# took me about 2 days to produce the first 2 riffs(with the same key)


# NEXT:
# 1)create_midi_lego_riff_combi -> 
    # try to use notes_df(output) from create_lego_riff_note_combi
        # build on top of create_midi_lego_riff_1file that will create midi files with different scales, keys, bpm in a loop with ost

# 2)convert midi directly to mp3(not important bc I can use format factory for now)
# try to do that manually(Clade Music_with_Code_03)

from music21 import stream
import pandas as pd
from midi2audio import FluidSynth
import fluidsynth
from typing import *
from pathlib import Path 
from mingus.core import scales, notes, intervals
from functools import partial
from mingus.containers import Note
import numpy as np
import inspect_py as inp
import pandas as pd
# notes.int_to_note()
# scales._Scale
# based on migus 0.6.1

OUTPUT_FOLDER = Path(r"C:\Users\Heng2020\OneDrive\D_Code\Python\Python Music\2024\01 Lego Riff Creation\lego_riff_creation\test_output")


# def convert_scale_degrees(
#     scale_degrees: List[int],
#     from_key: str,
#     from_scale_type: ScaleType,
#     to_key: str,
#     to_scale_type: ScaleType
# ) -> List[str]:
    
#     # Helper function to get the appropriate scale object
#     def get_scale(key: str, scale_type: ScaleType) -> scales._Scale:
#         if isinstance(scale_type, scales._Scale):
#             return scale_type(key)
#         elif isinstance(scale_type, str):
#             scale_type = scale_type.lower()
#             if scale_type == "major":
#                 return scales.Major(key)
#             elif scale_type in ["minor", "natural minor"]:
#                 return scales.NaturalMinor(key)
#             else:
#                 raise ValueError(f"Unsupported scale type string: {scale_type}")
#         else:
#             raise TypeError(f"Unsupported scale type: {type(scale_type)}")

#     # Create the scales
#     from_scale = get_scale(from_key, from_scale_type)
#     to_scale = get_scale(to_key, to_scale_type)
    
#     # Get the notes for the given scale degrees in the original scale
#     original_notes = [from_scale.get_note(degree - 1) for degree in scale_degrees]
    
#     # Find the interval between the two keys
#     interval = intervals.determine(from_key, to_key)
    
#     # Transpose each note
#     transposed_notes = [notes.augment(note, interval) for note in original_notes]
    
#     # Convert transposed notes to scale degrees in the new scale
#     new_scale_degrees = [to_scale.determine(note)[0] + 1 for note in transposed_notes]
    
#     return transposed_notes

# def convert_scale_degrees(
#     scale_degrees: List[int],
#     from_key: str,
#     from_scale_type: ScaleType,
#     to_key: str,
#     to_scale_type: ScaleType,
#     octave: int = 3
# ) -> List[str]:
#     def get_scale(key: str, scale_type: ScaleType) -> scales._Scale:
#         if isinstance(scale_type, scales.Scale):
#             return scale_type
#         elif isinstance(scale_type, type) and issubclass(scale_type, scales.Scale):
#             return scale_type(key)
#         elif isinstance(scale_type, str):
#             scale_type = scale_type.lower()
#             if scale_type == "major":
#                 return scales.Major(key)
#             elif scale_type in ["minor", "natural minor"]:
#                 return scales.NaturalMinor(key)
#             else:
#                 raise ValueError(f"Unsupported scale type string: {scale_type}")
#         else:
#             raise TypeError(f"Unsupported scale type: {type(scale_type)}")

#     from_scale = get_scale(from_key, from_scale_type)
#     to_scale = get_scale(to_key, to_scale_type)
    
#     # Get the notes for the given scale degrees in the original scale
#     original_notes = [Note(from_scale.get_notes(degree - 1), octave) for degree in scale_degrees]
    
#     # Adjust octaves for notes above the reference octave
#     for i, note in enumerate(original_notes):
#         if i > 0 and note.name < original_notes[i-1].name:
#             note.octave += 1

#     # Find the interval between the two keys
#     interval = intervals.determine(from_key, to_key)
    
#     # Transpose each note
#     # transposed_notes is the Note object
#     transposed_notes = [Note(notes.augment(note.name, interval), note.octave) for note in original_notes]
#     transposed_notes_str = [f"{notes.augment(note.name, interval)}{note.octave}" for note in original_notes]
    
#     # Convert transposed notes to scale degrees in the new scale
#     new_scale_degrees = [to_scale.determine(note.name)[0] + 1 for note in transposed_notes]
    
#     return transposed_notes_str


####################################

def test_create_lego_riff_note_combi():
    input_test_list = [None]*100
    input_test_list[0] = {
        "out_filename": OUTPUT_FOLDER / "Dawn_create_1_file_240bpm_v01.mid"
        ,"lego_block_num": [2,3,2,1,2]
        ,"note_lengths": [1,0.5,0.5,1,1]
        ,"directions": "up"
        ,"bpm": [240,200]
        ,"n": 7
        ,"root_degree": "max"
        ,"longer_last_note": 1
        
    }

    input_test_list[1] = {
        "out_filename": OUTPUT_FOLDER / "Dusk_create_1_file_200bpm_v01.mid"
        ,"lego_block_num": [3,2,1]
        ,"note_lengths":  [0.75,0.25,1]
        ,"bpm": 200
        ,"n": 6
        ,"root_degree": 2
        ,"longer_last_note": 1
        
    }
    for info_dict in input_test_list:
        if info_dict:
            riff_info_only = {key: info_dict[key] for key in info_dict.keys() if key not in ["out_filename","note_lengths","bpm","longer_last_note"] }
            riff_notes = create_lego_riff_note_combi(**riff_info_only)
            

def test_midi_to_audio():
    import py_string_tool as pst
    # create_midi_with_notes('EDC_music21.mid', ['E4', 'D4', 'C4'])
    midi_path = r"C:\Users\Heng2020\OneDrive\D_Code\Python\Python Music\2024\01 Lego Riff Creation\lego_riff_creation\test_output\EDC_music21.mid"
    
    midi_path02 = pst.replace_backslash(midi_path)
    import os
    if os.path.exists(midi_path):
        print(f"Midi path exist")
    
    if os.path.exists(midi_path02):
        print(f"Midi path exist")
    fs = FluidSynth()
    fs.midi_to_audio(midi_path, 'EDC_music21.wav')
    # FluidSynth().play_midi(midi_path)

def test_make_num_seq():
    import numpy as np
    import inspect_py as inp
    block01 = [2,3,2,1,2]
    expect01 = [ 2,  3,  2,  1,  2,  3,  4,  3,  2,  3,  4,  5,  4,  3,  4,  5,  6,
        5,  4,  5,  6,  7,  6,  5,  6,  7,  8,  7,  6,  7,  8,  9,  8,  7,
        8,  9, 10,  9,  8,  9]
    actual01 = make_num_seq(block01,7,as_np=False)
    actual02 = make_num_seq(block01,7,as_np=True)
    expect02 = np.array(expect01)
    assert actual01 == expect01, inp.assert_message(actual01, expect01)
    assert np.array_equal(actual02 ,expect02), inp.assert_message(actual02, expect02)

def test_convert_num_to_scale():
    scale_degrees01 = [ 2,  3,  2,  1,  2,  3,  4,  3,  2,  3,  4,  5,  4,  3,  4,  5,  6,
        5,  4,  5,  6,  7,  6,  5,  6,  7,  8,  7,  6,  7,  8,  9,  8,  7,
        8,  9, 10,  9,  8,  9]
    key01 = "C"
    octave01 = 4
    actual01 = convert_num_to_scale(scale_degrees01)
    print(actual01)
    expect01 = ["D4", " E4", " D4", " C4", " D4", " E4", " F4", " E4", " D4", " E4", " F4", " G4", " F4", " E4", " F4", " G4", " A4", " G4", " F4", " G4", " A4", " B4", " A4", " G4", " A4", " B4", " C5", " B4", " A4", " B4", " C5", " D5", " C5", " B4", " C5", " D5", " E5", " D5", " C5", " D5"]
    assert actual01 == expect01, inp.assert_message(actual01,expect01)
def test_make_num_degree_down():
    block01 = [3,2,1]
    actual01 = make_num_degree_down(block01,6,as_np=False)
    actual02 = make_num_degree_down(block01,6,as_np=True)
    actual03 = make_num_degree_down(block01,6,as_np=False,as_positive=False,root_degree="max")
    actual04 = make_num_degree_down(block01,6,as_np=True,as_positive=False,root_degree="max")


    expect01 = [8, 7, 6, 7, 6, 5, 6, 5, 4, 5, 4, 3, 4, 3, 2, 3, 2, 1, 2, 1, 0]
    expect02 = np.array(expect01)
    expect03 = [0, -1, -2, -1, -2, -3, -2, -3, -4, -3, -4, -5, -4, -5, -6, -5, -6, -7, -6, -7, -8]
    expect04 = np.array(expect03)

    # print(actual03)
    # print(actual04)

    assert actual01 == expect01, inp.assert_message(actual01,expect01)
    assert np.array_equal(actual02 ,expect02), inp.assert_message(actual02, expect02)

    assert actual03 == expect03, inp.assert_message(actual03,expect03)
    assert np.array_equal(actual04 ,expect04), inp.assert_message(actual04, expect04)
    
def test_create_midi_lego_riff_1file():
    input_test_list = [None]*100
    input_test_list[0] = {
        "out_filename": OUTPUT_FOLDER / "Dawn_create_1_file_240bpm_v01.mid"
        ,"lego_block_num": [2,3,2,1,2]
        ,"note_lengths": [1,0.5,0.5,1,1]
        ,"direction": "up"
        ,"bpm": 240
        ,"n": 7
        ,"key": "C"
        ,"scale_type": "Major"
        ,"root_degree": "max"
        ,"longer_last_note": 1
        
    }

    input_test_list[1] = {
        "out_filename": OUTPUT_FOLDER / "Dusk_create_1_file_200bpm_v01.mid"
        ,"lego_block_num": [3,2,1]
        ,"note_lengths":  [0.75,0.25,1]
        ,"direction": "down"
        ,"bpm": 200
        ,"n": 6
        ,"key": "C"
        ,"scale_type": "Minor"
        ,"root_degree": 2
        ,"longer_last_note": 1
        
    }
    for info_dict in input_test_list:
        if info_dict:
            riff_info_only = {key: info_dict[key] for key in info_dict.keys() if key not in ["out_filename","note_lengths","bpm","longer_last_note"] }
            riff_notes = create_lego_riff_note(**riff_info_only)
            create_midi_lego_riff_1file(**info_dict)


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
    create_riff03()
    create_riff02()
    create_block02()
    # create_riff01()
    # create_block01()

def main_test():
    test_create_lego_riff_note_combi()
    test_create_midi_lego_riff_1file()
    test_make_num_degree_down()
    # test_convert_num_to_scale()
    # test_make_num_seq()
    # test_midi_to_audio()

def main():
    main_test()
    main_real_test()
    
if __name__ == '__main__':
    main()