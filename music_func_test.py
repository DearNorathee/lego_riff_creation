from typing import List, Union, Dict, Literal
import pandas as pd
from pathlib import Path
from mingus.core import scales, notes, intervals
from music_func import *
import inspect_py as inp

OUTPUT_FOLDER = Path(r"C:\Users\Heng2020\OneDrive\D_Code\Python\Python Music\2024\01 Lego Riff Creation\lego_riff_creation\test_output")

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
    from midi2audio import FluidSynth
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

def main_test():
    test_create_lego_riff_note_combi()
    test_create_midi_lego_riff_1file()
    test_make_num_degree_down()
    # test_convert_num_to_scale()
    # test_make_num_seq()
    # test_midi_to_audio()

def main():
    main_test()
    
if __name__ == '__main__':
    main()