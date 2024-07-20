# use env => latest pandas
from music21 import *
import pandas as pd
from midi2audio import FluidSynth
import fluidsynth
from typing import *
from pathlib import Path
from mingus.core import scales, notes, intervals


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