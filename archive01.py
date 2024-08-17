
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