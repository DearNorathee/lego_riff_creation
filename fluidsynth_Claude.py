import fluidsynth
import numpy as np
from pydub import AudioSegment
import io

# Initialize FluidSynth
fs = fluidsynth.Synth()
fs.start()

# Load a SoundFont (you'll need to download a SoundFont file)
sfid = fs.sfload("path/to/your/soundfont.sf2")
fs.program_select(0, sfid, 0, 0)

# Create a simple melody (middle C, D, E, F, G)
notes = [60, 62, 64, 65, 67]
duration = 1  # seconds per note

# Generate audio
audio = []
for note in notes:
    fs.noteon(0, note, 100)
    samples = fs.get_samples(int(44100 * duration))
    audio.extend(samples)
    fs.noteoff(0, note)

audio = np.array(audio).reshape(-1, 2)

# Convert to 16-bit PCM
audio = (audio * 32767).astype(np.int16)

# Use pydub to create an AudioSegment
audio_segment = AudioSegment(
    audio.tobytes(), 
    frame_rate=44100,
    sample_width=2, 
    channels=2
)

# Export as MP3
audio_segment.export("piano_melody.mp3", format="mp3")

# Clean up
fs.delete()