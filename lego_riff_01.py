# use env => latest pandas
from music21 import *
import pandas as pd
from midi2audio import FluidSynth
import fluidsynth
from typing import *
from pathlib import Path
from mingus.core import scales, notes, intervals
from functools import partial
from mingus.containers import Note

# scales._Scale
ScaleType = Union[str, scales._Scale]

def create_midi(out_filename:Union[Path,str], note_names:List[str], note_lengths:Union[None,List[float]]) -> None:
    """
    Create a MIDI file with the specified note names and save it to the given filename.
    
    Parameters:
    filename (str): The name of the MIDI file to be created.
    note_names (list): List of strings representing the names of the notes to be added to the MIDI file.
    
    note_lengths: 1 represent quater note

    Returns:
    None
    """
    # Create a stream
    s = stream.Stream()

    # Add notes to the stream
    for note_name in note_names:
        n = note.Note(note_name)
        n.quarterLength = 1  # Each note lasts for one quarter note
        s.append(n)

    # Create a MIDI file
    mf = midi.translate.streamToMidiFile(s)
    
    # Write the MIDI file
    mf.open(out_filename, 'wb')
    mf.write()
    mf.close()

    # print(f"MIDI file '{filename}' has been created with notes {', '.join(note_names)}.")

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

def convert_scale_degrees(
    scale_degrees: List[int],
    from_key: str,
    from_scale_type: ScaleType,
    to_key: str,
    to_scale_type: ScaleType,
    octave: int = 4
) -> List[str]:
    def get_scale(key: str, scale_type: ScaleType) -> scales.Scale:
        if isinstance(scale_type, scales.Scale):
            return scale_type
        elif isinstance(scale_type, type) and issubclass(scale_type, scales.Scale):
            return scale_type(key)
        elif isinstance(scale_type, str):
            scale_type = scale_type.lower()
            if scale_type == "major":
                return scales.Major(key)
            elif scale_type in ["minor", "natural minor"]:
                return scales.NaturalMinor(key)
            else:
                raise ValueError(f"Unsupported scale type string: {scale_type}")
        else:
            raise TypeError(f"Unsupported scale type: {type(scale_type)}")

    from_scale = get_scale(from_key, from_scale_type)
    to_scale = get_scale(to_key, to_scale_type)
    
    # Get the notes for the given scale degrees in the original scale
    original_notes = [Note(from_scale.get_note(degree - 1), octave) for degree in scale_degrees]
    
    # Adjust octaves for notes above the reference octave
    for i, note in enumerate(original_notes):
        if i > 0 and note.name < original_notes[i-1].name:
            note.octave += 1

    # Find the interval between the two keys
    interval = intervals.determine(from_key, to_key)
    
    # Transpose each note
    # transposed_notes is the Note object
    transposed_notes = [Note(notes.augment(note.name, interval), note.octave) for note in original_notes]
    transposed_notes_str = [f"{notes.augment(note.name, interval)}{note.octave}" for note in original_notes]
    
    # Convert transposed notes to scale degrees in the new scale
    new_scale_degrees = [to_scale.determine(note.name)[0] + 1 for note in transposed_notes]
    
    return transposed_notes_str


def convert_num_to_scale(scale_degrees:List[int], 
                         key: str, 
                         scale_type: ScaleType) -> List[str]:
    convert_num_to_scale_partial = partial(convert_scale_degrees,from_key = "C", from_scale_type="major", to_scale_type=scale_type,to_key = key)
    return convert_num_to_scale_partial

def make_num_seq(num_block:List[int],n:int = 7, increment:int = 1,as_np:bool=False) -> List[int]:
    """
    Generates a sequence of numbers by repeating the given `num_block` list `n` times and incrementing each element by `increment`.
    
    Args:
        num_block (List[int]): The initial list of numbers.
        n (int, optional): The number of times to repeat the `num_block` list. Defaults to 7.
        increment (int, optional): The amount to increment each element by. Defaults to 1.
        as_np (bool, optional): Whether to return the result as a NumPy array. Defaults to False.
    
    Returns:
        List[int] or np.ndarray: The generated sequence of numbers. If `as_np` is True, returns a NumPy array.
    """
    import numpy as np

    base_np_array = np.array(num_block)
    out_array = np.array(num_block)
    for i in range(1,n+1):
        np_array = base_np_array + i*increment
        out_array = np.append(out_array, np_array)
    out_list = out_array.tolist()

    if as_np:
        return out_array
    else:
        return out_list



####################################


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

def main():
    test_make_num_seq()
    test_midi_to_audio()

if __name__ == '__main__':
    main()