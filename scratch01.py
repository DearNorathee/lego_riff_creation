from music21 import *
import pandas as pd
from midi2audio import FluidSynth
import fluidsynth
from mingus.midi import *
from typing import *
from pathlib import Path
from mingus.core import scales, notes, intervals
from functools import partial
from mingus.containers import Note
# print(type(scales.Major))
import mingus
print(mingus.__version__)
# from mingus.core import scales, notes, intervals
print(scales.Ionian("C").ascending())
fluidsynth.play_Note("C",4)
