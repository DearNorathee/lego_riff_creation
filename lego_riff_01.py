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

ScaleType = Union[scales._Scale,Literal["Major","Minor","Natural minor","Ionian","Dorian","Phrygian","Lydian","Mixolydian","Aeolian","Locrian","Harmonic minor","Melodic minor","Whole tone","Chromatic"]]
OUTPUT_FOLDER = Path(r"C:\Users\Heng2020\OneDrive\D_Code\Python\Python Music\2024\01 Lego Riff Creation\lego_riff_creation\test_output")


def create_lego_riff_note_combi(
    # out_prefixname:Union[Path,str]
    lego_block_num: List[int]
    # ,note_lengths: List[int]
    ,directions: Union[List[str],Literal["up","down"]] = ["up","down"]
    # ,bpms: Union[int,List[int]] = 120
    ,n: Union[int,  Dict[Literal["up","down"],int]  ] = 7
    ,key_s: Union[str,List[str]] = ["C", "D", "E", "F", "G", "A", "B", "C#", "Eb", "F#", "Ab", "Bb"]
    ,scale_types: Union[ScaleType, List[ScaleType]] = ["Major","Natural minor","Dorian","Phrygian","Lydian","Mixolydian","Locrian"]
    ,root_degree: Union[Literal["max","min"],int] = "max"
    ,octaves: Union[int,List[int]] = 3
    ,longer_last_note:Union[bool,int] = 1
    ) -> pd.DataFrame:
    # took about 1 hr(including testing)
    import pandas as pd
    """
    signature function

    build on top of create_lego_riff_note to loop through different parameters and create a combination of midi files
    n can be int or dict  with key "up" and "down" eg n = {"up":6,"down":7}

    output pd.DataFrame {'direction','scale_type','key','octave','notes'}

    """
    # has error with D# as key Gb

    directions_in = list(directions) if isinstance(directions,list) else [directions]
    scale_types_in = list(scale_types) if isinstance(scale_types,list) else [scale_types]
    key_s_in = list(key_s) if isinstance(key_s,list) else [key_s]

    # has error with D# as key Gb, so I want to support D#, Gb as input
    # for other key like Db, G#, A# I would replace them(not to sure that it would throw an error but just replace for safety)
    REPLACE_KEY_DICT = {
        'D#': 'Eb',
        'Gb': 'F#',
        'Db': 'C#',
        'G#': 'Ab',
        'A#': 'Bb'
    }

    for i, key in enumerate(key_s_in):
        if key in REPLACE_KEY_DICT.keys():
            key_s_in[i] = REPLACE_KEY_DICT[key]


    octaves_in = list(octaves) if isinstance(octaves,list) else [octaves]

    out_df = pd.DataFrame(columns = ["direction","scale_type","key","octave","notes"])
    results = []

    for direction in directions_in:
        for scale_type in scale_types_in:
            for key in key_s_in:
                for octave in octaves_in:
                    if isinstance(n,int):
                        curr_riff_notes = create_lego_riff_note(
                                            lego_block_num = lego_block_num
                                            ,direction = direction
                                            ,n = n
                                            ,key = key
                                            ,scale_type = scale_type
                                            ,root_degree = root_degree
                                            ,octave = octave
                                            )
                    elif isinstance(n,dict):
                        curr_riff_notes = create_lego_riff_note(
                                            lego_block_num = lego_block_num
                                            ,direction = direction
                                            ,n = n[direction]
                                            ,key = key
                                            ,scale_type = scale_type
                                            ,root_degree = root_degree
                                            ,octave = octave
                                            )
                    curr_dict = {
                        "direction": direction,
                        "scale_type": scale_type,
                        "key": key,
                        "octave": octave,
                        "notes": curr_riff_notes
                        }
                    results.append(curr_dict)

    out_df = pd.DataFrame(results)
    return out_df
     


def create_midi_lego_riff_combi(
    out_prefixname:Union[Path,str]
    ,lego_block_num: List[int]
    ,note_lengths: List[int]
    ,directions: Union[List[str],Literal["up","down"]] = ["up","down"]
    ,bpms: Union[int,List[int]] = 120
    ,n: Union[int,  Dict[Literal["up","down"],int]  ] = 7
    ,key_s: Union[str,List[str]] = ["C", "D", "E", "F", "G", "A", "B", "C#", "Eb", "F#", "Ab", "Bb"]
    ,scale_types: Union[ScaleType, List[ScaleType]] = ["Major","Natural minor","Dorian","Phrygian","Lydian","Mixolydian","Locrian"]
    ,root_degree: Union[Literal["max","min"],int] = "max"
    ,octaves: Union[int,List[int]] = 3
    ,longer_last_note:Union[bool,int] = 1
    ) -> None:

    # has error with D# as key Gb

    """
    out_prefixname is without ".mid"
    signature function

    build on top of create_midi_lego_riff_1file to loop through different parameters and create a combination of midi files
    n can be int or dict  with key "up" and "down" eg n = {"up":6,"down":7}

    """
    import os_toolkit as ost
    # convert single value all into list to support both
    directions_in = list(directions) if isinstance(directions,list) else [directions]
    scale_types_in = list(scale_types) if isinstance(scale_types,list) else [scale_types]
    key_s_in = list(key_s) if isinstance(key_s,list) else [key_s]
    octaves_in = list(octaves) if isinstance(octaves,list) else [octaves]
    bpms_in = list(bpms) if isinstance(directions,list) else [bpms]

    pass
def create_midi(out_filename: Union[Path, str], 
                note_names: List[str], 
                note_lengths: Union[None, List[float]] = None,
                bpm: float = 120) -> None:
    from music21 import stream, note, midi, tempo
    # high tested
    """
    Create a MIDI file with the specified note names, lengths, and tempo, then save it to the given filename.
    
    Parameters:
    out_filename (Union[Path, str]): The name of the MIDI file to be created.
    note_names (List[str]): List of strings representing the names of the notes to be added to the MIDI file.
    note_lengths (Union[None, List[float]]): List of note durations. 1.0 represents a quarter note. If None, all notes are quarter notes.
    bpm (float): Tempo in beats per minute. Defaults to 120 BPM.
    
    Returns:
    None
    """
    # Create a stream
    s = stream.Stream()

    # Add tempo marking
    t = tempo.MetronomeMark(number=bpm)
    s.append(t)

    # Add notes to the stream
    for i, note_name in enumerate(note_names):
        n = note.Note(note_name)
        if note_lengths is None:
            n.quarterLength = 1  # Each note lasts for one quarter note
        else:
            n.quarterLength = note_lengths[i]
        s.append(n)

    # Create a MIDI file
    mf = midi.translate.streamToMidiFile(s)
    
    # Write the MIDI file
    mf.open(str(out_filename), 'wb')
    mf.write()
    mf.close()

def create_midi_repeate_tempo(
        out_filename:Union[Path,str], 
        note_names:List[str], 
        note_lengths:Union[None,List[float]],
        bpm:float = 120,
        longer_last_note:Union[int,bool] =1,
        ) -> None:
    # high tested
    """
    do similar thing to create_midi but note_lengths is smarter because it would take only the block of note_lengths, and assume to have the same tempo the whole time
    longer_last_note will extend the lastnote for a bit
    """
    from music21 import stream,note,midi,tempo
    s = stream.Stream()

    t = tempo.MetronomeMark(number=bpm)
    s.append(t)
    # if len(note_names) is not divisible by len(note_lengths) it would raise an error
    if len(note_names) % len(note_lengths) != 0:
        raise Exception(f"len(note_names) must be divisible by len(note_lengths), but got len(note_names)={len(note_names)} and len(note_lengths)={len(note_lengths)}")

    n_repeat = len(note_names) // len(note_lengths)
    note_lengths_repeat = note_lengths * n_repeat

    # Add notes to the stream
    for i, note_name in enumerate(note_names):
        n = note.Note(note_name)
        if note_lengths is None:
            n.quarterLength = 1  # Each note lasts for one quarter note
        else:
            if (i == len(note_names) - 1):
                n.quarterLength = note_lengths_repeat[i] + longer_last_note
            else:
                n.quarterLength = note_lengths_repeat[i]
        s.append(n)

    # Create a MIDI file
    mf = midi.translate.streamToMidiFile(s)
    
    # Write the MIDI file
    mf.open(out_filename, 'wb')
    mf.write()
    mf.close()

    pass

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

def _get_scale(key: str, scale_type: ScaleType) -> scales._Scale:
        # Helper function to get the appropriate scale object
        scale_dict = {
            "major": scales.Major,
            "minor": scales.NaturalMinor,
            "natural minor": scales.NaturalMinor,
            "ionian": scales.Ionian,
            "dorian": scales.Dorian,
            "phrygian": scales.Phrygian,
            "lydian": scales.Lydian,
            "mixolydian": scales.Mixolydian,
            "aeolian": scales.Aeolian,
            "locrian": scales.Locrian,
            # Add more scales as needed
            "harmonic minor": scales.HarmonicMinor,
            "melodic minor": scales.MelodicMinor,
            "whole tone": scales.WholeTone,
            "chromatic": scales.Chromatic,
            # "major pentatonic": scales.,

        }
        # no pentatonic scale in based on migus 0.6.1
        
        if isinstance(scale_type, scales._Scale):
            return scale_type
        elif isinstance(scale_type, type) and issubclass(scale_type, scales._Scale):
            return scale_type(key)
        elif isinstance(scale_type, str):
            scale_type = scale_type.lower()
            if scale_type in scale_dict.keys():
                return scale_dict[scale_type](key)
            else:
                raise ValueError(f"Unsupported scale type string: {scale_type}")
        else:
            raise TypeError(f"Unsupported scale type: {type(scale_type)}")

def convert_num_to_scale(scale_degrees:List[int], 
                         key: str = "C", 
                         octave: int = 4,
                         scale_type: ScaleType = scales.Major,
                         out_as_str:bool = True
                         ) -> Union[List[str], List[Note]]:
    scales_obj = _get_scale(key, scale_type)
    scales = scales_obj.ascending()
    # shift index by 1
    #  0 & 1 would be the same note
    # -1 -2 will refer to the note below
    scales = [scales[0]] + scales
    scales_notes = []

    for degree in scale_degrees:
        curr_octave = degree // 7
        if degree % 7 == 0:
            curr_degree = 7
            add_to_octave = 0
        else:
            curr_degree = degree % 7
            add_to_octave = degree // 7

        curr_note = Note(scales[curr_degree],octave+add_to_octave )
        scales_notes.append(curr_note)

    scales_notes_str = [f"{note.name}{note.octave}" for note in scales_notes]

    if out_as_str:
        return scales_notes_str
    else:
        return scales_notes
    

def make_num_seq(num_block:List[int],n:int = 7, increment:int = 1,as_np:bool=False) -> Union[List[int], np.ndarray[np.int_]]:


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

def make_num_degree_down(
        num_block:List[int],
        n:int = 7, 
        increment:int = -1,
        as_np:bool=False,
        as_positive:bool = True,
        root_degree:Union[Literal["max","min"],int] = "max"
        )-> Union[List[int], np.ndarray[np.int_]]:
    """ 
    the reason I have to write this function because 0,1 will be interpreted as the root note and only -1 means the 1 step below root note
    it's only for my lib and it won't be useful somewhere else

    as_positive will +8 to make the num_degree become positive

    """
    if root_degree in ["max"]:
        root_num = max(num_block)
    elif root_degree in ["min"]:
        root_num = min(num_block)
    elif isinstance(root_degree, int):
        root_num = root_degree

    num_block_negative = [num-root_num for num in num_block]

    base_np_array = np.array(num_block_negative)
    out_array_negative = np.array(num_block_negative)
    for i in range(1,n+1):
        np_array = base_np_array + i*increment
        out_array_negative = np.append(out_array_negative, np_array)
    
    out_array_positive = out_array_negative + 8
    out_list_negative = out_array_negative.tolist()
    out_list_positive = out_array_positive.tolist()

    if as_np:
        if as_positive:
            return out_array_positive
        else:
            return out_array_negative
    else:
        if as_positive:
            return out_list_positive
        else:
            return out_list_negative

def create_lego_riff_note(
        lego_block_num: List[int]
        ,direction: Literal["up","down"]
        ,n: int = 7
        ,key:str = "C"
        ,scale_type:ScaleType = "Major"
        ,root_degree:Union[Literal["max","min"],int] = "max"
        ,octave:int = 3
        ,out_as_str:bool = True
        ) -> Union[List[Note],List[str]]:
    # medium tested
    """
    Generate a musical riff based on a sequence of Lego block numbers.

    Args:
        lego_block_num (List[int]): A list of Lego block numbers.
        direction (Literal["up","down"]): The direction of the riff. "up" for ascending and "down" for descending.
        n (int, optional): The number of notes in the riff. Defaults to 7.
        key (str, optional): The key of the riff. Defaults to "C".
        scale_type (ScaleType, optional): The type of scale to use. Defaults to "Major".
        root_degree (Union[Literal["max","min"],int], optional): The root degree of the scale. Defaults to "max".
        out_as_str (bool, optional): Whether to output the riff as a list of strings or a list of Note objects. Defaults to True.

    Returns:
        Union[List[Note],List[str]]: The generated riff as a list of Note objects or a list of strings.

    Raises:
        ValueError: If the direction is neither "up" nor "down".
    """
    
    if direction.lower() in ["up"]:
        scale_degrees = make_num_seq(lego_block_num, n=n, increment=1)
    elif direction.lower() in ["down"]:
        scale_degrees = make_num_degree_down(lego_block_num, n=n, increment=-1,root_degree=root_degree)
    else:
        raise ValueError("direction must be either 'up' or 'down'")
    
    riff_notes_obj = convert_num_to_scale(scale_degrees,scale_type=scale_type,key=key,out_as_str=False,octave=octave)
    riff_notes_str = convert_num_to_scale(scale_degrees,scale_type=scale_type,key=key,out_as_str=True,octave=octave)

    if out_as_str:
        return riff_notes_str
    else:
        return riff_notes_obj

def create_midi_lego_riff_1file(
    out_filename:Union[Path,str]
    ,lego_block_num: List[int]
    ,note_lengths: List[int]
    ,direction: Literal["up","down"]
    ,bpm:int = 120
    ,n: int = 7
    ,key:str = "C"
    ,scale_type:ScaleType = "Major"
    ,root_degree:Union[Literal["max","min"],int] = "max"
    ,longer_last_note:Union[bool,int] = 1
    ) -> None:

    """
    signature function
    """

    # medium tested
    riff_notes = create_lego_riff_note(
        lego_block_num = lego_block_num
        ,direction = direction
        ,n = n
        ,key = key
        ,scale_type = scale_type
        ,root_degree = root_degree
        ,out_as_str = True
        )
    
    create_midi_repeate_tempo(out_filename,riff_notes,note_lengths,bpm=bpm,longer_last_note=longer_last_note)


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