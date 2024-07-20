# use env => latest pandas
from music21 import *
import pandas as pd
from midi2audio import FluidSynth
import fluidsynth
from typing import *
from pathlib import Path
from mingus.core import scales, notes, intervals
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

def convert_scale_degrees(
    scale_degrees: List[int],
    from_key: str,
    from_scale_type: ScaleType,
    to_key: str,
    to_scale_type: ScaleType
) -> List[str]:
    
    # Helper function to get the appropriate scale object
    def get_scale(key: str, scale_type: ScaleType) -> scales._Scale:
        if isinstance(scale_type, scales.Scale):
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

    # Create the scales
    from_scale = get_scale(from_key, from_scale_type)
    to_scale = get_scale(to_key, to_scale_type)
    
    # Get the notes for the given scale degrees in the original scale
    original_notes = [from_scale.get_note(degree - 1) for degree in scale_degrees]
    
    # Find the interval between the two keys
    interval = intervals.determine(from_key, to_key)
    
    # Transpose each note
    transposed_notes = [notes.augment(note, interval) for note in original_notes]
    
    # Convert transposed notes to scale degrees in the new scale
    new_scale_degrees = [to_scale.determine(note)[0] + 1 for note in transposed_notes]
    
    return transposed_notes, new_scale_degrees



def test_create_midi_with_notes():
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
    fs.midi_to_audio(midi_path02, r"C:\Users\Heng2020\OneDrive\D_Code\Python\Python Music\2024\01 Lego Riff Creation\lego_riff_creation\test_output\EDC_music21.wav")
    # FluidSynth().play_midi(midi_path)

def main():
    test_create_midi_with_notes()

if __name__ == '__main__':
    main()