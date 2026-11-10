# digital_neuropath

Blind per-neuron scoring of 3D confocal stacks, with optional 3D cilia/centrosome detection.

## Repository

- `README.md` — this file
- `score_neurons.py` — the script you edit and run: where your files are and your choices, on top
- `digital_neuropath/` — the modules (census, scoring, summaries, 3D cilia)
- `output/` — what the tool writes (answers, crops, result tables); stays out of git

## Install

Requires Python 3.10 or newer.

```bash
git clone https://github.com/atudoras/digital_neuropath
cd digital_neuropath
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

Leave your files where they are: list their paths in `score_neurons.py` under INPUT
(`FOLDER` once, then `f"{FOLDER}/<file name>"`), set the channels and stains, then:

```bash
python3 score_neurons.py
```

`IMAGES`, `NEURON_MASKS` and `NUCLEI_MASKS` are lists of files paired by position: the 1st mask
goes with the 1st image. Images are `.tif` (OME or ImageJ, voxel size read from the file) or
`.lif`; masks are labelled `.tif` or NeuroVIT `.cmap`. Each image file is one case. Close a window
to stop; the same command resumes.

## Modules

| Module | What you do |
|---|---|
| Neurons | click each neuron once on its widest plane; close the window, type `y` in Terminal |
| Refinement | click a neuron to include/exclude (orange = suggested exclusion); close the window when done |
| Scoring | `1`–`k` your categories, then unsure and not scoreable · `n` note · `s` skip · `b` back · `m` MIP/planes · `o` outlines; contrast you set is kept for the next neurons |

## Outputs

`output/results.csv` — one row per neuron: image, case, position, every stain's call and note. Everything else in `output/` is the tool's working state; leave it in place.
