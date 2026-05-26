import os
import traceback

import av
import librosa
import numpy as np


def wav2(i, o, format):
    # PyAV>=12 expects mode "r" / "w" (not "rb" / "wb").
    inp = av.open(i, "r")
    if format == "m4a":
        format = "mp4"
    out = av.open(o, "w", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "mp4":
        format = "aac"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


def audio2(i, o, format, sr):
    inp = av.open(i, "r")
    out = av.open(o, "w", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "f32le":
        format = "pcm_f32le"

    ostream = out.add_stream(format, channels=1)
    ostream.sample_rate = sr

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    out.close()
    inp.close()


def load_audio(file, sr):
    """Decode to mono float32 at ``sr``.

    Older implementation round-tripped through PyAV into a raw PCM BytesIO buffer. That path
    bitrots on newer PyAV (modes, ``channels`` on ``add_stream``) and mishandled failures via an
    ``except AttributeError`` branch that assumed ``file`` was a ``(native_sr, samples)`` tuple.
    ``librosa.load`` matches VC inference needs and stays compatible across PyAV versions.
    """
    if not os.path.exists(file):
        raise RuntimeError(
            "You input a wrong audio path that does not exists, please fix it!"
        )
    try:
        audio, _ = librosa.load(str(file), sr=sr, mono=True)
        return np.asarray(audio, dtype=np.float32).flatten()

    except Exception:
        raise RuntimeError(traceback.format_exc())
