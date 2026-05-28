# Retrieval-based-Voice-Conversion

Programmatic voice conversion inference library based on RVC (Retrieval-based Voice Conversion).

## Installation

```sh
pip install git+https://github.com/RVC-Project/Retrieval-based-Voice-Conversion
```

Requires Python 3.10 or newer (3.12 recommended).

With [uv](https://docs.astral.sh/uv/) from a checkout:

```sh
uv sync
uv run python your_script.py
```

## Required assets

Set these environment variables (or use a `.env` file with `python-dotenv`):

| Variable | Description |
|----------|-------------|
| `weight_root` | Directory containing `.pth` voice models |
| `index_root` | Directory containing `.index` feature index files |
| `hubert_path` | Path to `hubert_base.pt` |
| `rmvpe_root` | Directory containing `rmvpe.pt` |

Example `.env`:

```env
weight_root=./assets/weights
index_root=./assets/indices
hubert_path=./assets/hubert_base.pt
rmvpe_root=./assets/rmvpe
```

## Usage

```python
from pathlib import Path

from dotenv import load_dotenv
from scipy.io import wavfile

from rvc import VC

load_dotenv()

vc = VC()
vc.get_vc("model.pth")

tgt_sr, audio_opt, times, err = vc.vc_inference(
    sid=0,
    input_audio_path=Path("input.wav"),
    f0_method="rmvpe",
    index_rate=0.75,
)

if err:
    raise RuntimeError(err)

wavfile.write("output.wav", tgt_sr, audio_opt)
```

### Batch inference

```python
for progress in vc.vc_multi(
    sid=0,
    paths=["a.wav", "b.wav"],
    opt_root="./output",
):
    print(progress)
```

### Parameters

`vc_inference` supports pitch and quality tuning:

- `f0_up_key` — semitone transpose (12 = one octave up)
- `f0_method` — `rmvpe`, `pm`, `harvest`, or `crepe`
- `index_file` / `index_rate` — feature retrieval index and blend ratio
- `filter_radius` — median filter on harvested pitch
- `resample_sr` — output sample rate (0 = model default)
- `rms_mix_rate` — volume envelope blend
- `protect` — protect voiceless consonants (lower = more protection)

## License

See [LICENSE](LICENSE).
