# EEG Spectrogram Generator

Python program for converting BrainVision EEG recordings into word-level spectrogram datasets aligned to spoken-word onset timing.

## Features

- YAML-based configuration via `config.yaml`
- BrainVision EEG loading from `.vhdr`, `.vmrk`, `.eeg`
- 60 Hz notch filtering
- Word-aligned EEG extraction
- `256 x 256` spectrogram generation
- Multiple latency conditions such as `0`, `100`, `300` ms
- Raw, log, normalized, and log-normalized output variants

## Project Layout

- `src/generate_eeg_spectrograms.py`: main generation script
- `config.yaml`: runtime configuration
- `Data/`: raw EEG input data (not tracked in git)
- `outputs/`: generated spectrograms and metadata (not tracked in git)

## Expected Input

The script expects BrainVision EEG files in this structure:

```text
Data/
  Raw_EEG/
    S01/
      S01.eeg
      S01.vhdr
      S01.vmrk
    S02/
      ...
  exp_audio_eeg_timing_info.csv
```

The timing CSV must include at least these columns:

- `stimulus`
- `interval_index`
- `word`
- `rounded_onset_sec`
- `rounded_duration_sec`

## Not Included In Git

The following folders are intentionally excluded from version control:

- `Data/`
- `outputs/`

## Installation

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

All settings are controlled in `config.yaml`.

Examples:

- Process all subjects: `subjects: all`
- Process selected subjects: `subjects: [S01, S02]`
- Process all EEG channels: `channels: all`
- Process selected channels: `channels: ["1", "2"]`

## Run

```bash
python src/generate_eeg_spectrograms.py
```

## Outputs

For each subject and word, the script creates latency-condition folders such as:

```text
outputs/word_spectrograms/S01/<word>/0/
outputs/word_spectrograms/S01/<word>/100/
outputs/word_spectrograms/S01/<word>/300/
```

Inside each latency folder, the following subfolders are created:

- `raw`
- `raw_normalized`
- `log_raw`
- `log_normalized`

Each output set also updates:

- `outputs/word_spectrograms/<subject>/metadata.csv`
- `outputs/word_spectrograms/summary.csv`
