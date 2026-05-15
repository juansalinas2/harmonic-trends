from utils import chords_to_tensor as ctt
import numpy as np
import wave


PC_TO_SEMITONE = np.arange(12)  # 0..11 relative to C

def midi_to_hz(midi: float, a4: float = 440.0) -> float:
    return a4 * (2.0 ** ((midi - 69.0) / 12.0))

def pcs_to_freqs(pcs, base_midi: int) -> list[float]:
    # pcs are integers 0..11, base_midi is midi number for pitch class 0 (C) at some octave
    return [midi_to_hz(base_midi + int(pc)) for pc in pcs]

def synth_chord(freqs, dur, sr=44100, amp=0.2):
    n = int(dur * sr)
    t = np.linspace(0, dur, n, endpoint=False)

    if len(freqs) == 0:
        return np.zeros(n, dtype=np.float32)

    # simple ADSR-ish envelope to avoid clicks
    attack = int(0.01 * sr)
    release = int(0.02 * sr)
    env = np.ones(n, dtype=np.float32)
    if attack > 0:
        env[:attack] = np.linspace(0, 1, attack, endpoint=False)
    if release > 0:
        env[-release:] = np.linspace(1, 0, release, endpoint=False)

    y = np.zeros(n, dtype=np.float32)
    for f in freqs:
        y += np.sin(2 * np.pi * f * t).astype(np.float32)

    y /= max(1, len(freqs))  # normalize by #partials
    y *= amp
    y *= env
    return y

def tensor_to_audio(x, frame_dur=0.35, sr=44100, a4=440.0):
    """
    x: np.ndarray or torch tensor, shape (3,12,L), values 0/1
    returns np.float32 mono audio
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = x.astype(np.uint8)
    assert x.ndim == 3 and x.shape[0] == 3 and x.shape[1] == 12

    L = x.shape[2]
    audio = []

    # choose octaves/roles (you can change these)
    base_core = 60   # C4
    base_ext  = 72   # C5
    base_bass = 36   # C2

    for t in range(L):
        core_pcs = np.flatnonzero(x[0, :, t]).tolist()
        ext_pcs  = np.flatnonzero(x[1, :, t]).tolist()
        bass_pcs = np.flatnonzero(x[2, :, t]).tolist()

        freqs = []
        freqs += pcs_to_freqs(core_pcs, base_core)
        freqs += pcs_to_freqs(ext_pcs,  base_ext)
        freqs += pcs_to_freqs(bass_pcs, base_bass)

        y = synth_chord(freqs, frame_dur, sr=sr, amp=0.25)
        audio.append(y)

    return np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32)

def write_wav(path, y, sr=44100):
    # y: float32 in [-1,1]
    y = np.clip(y, -1.0, 1.0)
    y16 = (y * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sr)
        wf.writeframes(y16.tobytes())



#########################################
# Example usage
#s = 'C Cmaj7 C7 F Fmin Fmin9 Bdim C Cmaj7'
#
#x = ctt.chord_string_to_rank3_tensor(s)   # your function -> (3,12,L)
#y = tensor_to_audio(x, frame_dur=0.8, sr=44100)
#write_wav("song.wav", y)