from __future__ import annotations

import csv
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from scipy import signal


DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass
class BrainVisionHeader:
    # Minimal subset of the BrainVision header that we need to locate and decode the EEG recording.
    data_file: str
    marker_file: str
    data_format: str
    data_orientation: str
    number_of_channels: int
    sampling_interval_usec: float
    channel_names: list[str]


@dataclass
class AppConfig:
    # Every runtime option is centralized here after being read from config.yaml.
    data_root: Path
    timing_csv: Path
    output_root: Path
    subjects: list[str] | None
    words: list[str] | None
    sample_rate: float | None
    freq_min: float
    freq_max: float
    target_size: int
    notch_freq: float
    notch_q: float
    word_window_sec: float
    extra_window_ms_options: list[int]
    preview_dpi: int
    channels: list[str] | None
    save_png: bool


class TerminalProgressBar:
    # Lightweight dependency-free progress bar for long-running terminal jobs.
    def __init__(self, total: int, label: str, width: int = 32) -> None:
        self.total = max(1, int(total))
        self.label = label
        self.width = max(10, int(width))
        self.current = 0

    def update(self, step: int = 1, suffix: str = "") -> None:
        self.current = min(self.total, self.current + step)
        filled = int(round(self.width * self.current / self.total))
        bar = "#" * filled + "-" * (self.width - filled)
        message = (
            f"\r[{bar}] {self.current}/{self.total} {self.label}"
            f"{' | ' + suffix if suffix else ''}"
        )
        print(message, end="", file=sys.stdout, flush=True)
        if self.current >= self.total:
            print(file=sys.stdout, flush=True)


def parse_selection_list(value: object, field_name: str) -> list[str] | None:
    # `all` and null are both treated as "use every available item".
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() == "all":
            return None
        return [value]
    if isinstance(value, list):
        if not value:
            raise ValueError(f"'{field_name}' in config.yaml must not be an empty list.")
        lowered = [str(item).lower() for item in value]
        if any(item == "all" for item in lowered):
            return None
        return [str(item) for item in value]
    raise ValueError(f"'{field_name}' in config.yaml must be 'all', a list, or null.")


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Keep all runtime knobs in YAML so the script can be launched from editors without CLI args
    # and so experiment settings can be changed without touching code.
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    subjects = parse_selection_list(raw_config.get("subjects"), "subjects")
    channels = parse_selection_list(raw_config.get("channels", ["1", "2"]), "channels")

    return AppConfig(
        data_root=Path(raw_config.get("data_root", "Data/Raw_EEG")),
        timing_csv=Path(raw_config.get("timing_csv", "Data/exp_audio_eeg_timing_info.csv")),
        output_root=Path(raw_config.get("output_root", "outputs/word_spectrograms")),
        subjects=subjects,
        words=parse_selection_list(raw_config.get("words"), "words"),
        sample_rate=raw_config.get("sample_rate"),
        freq_min=float(raw_config.get("freq_min", 0.1)),
        freq_max=float(raw_config.get("freq_max", 50.0)),
        target_size=int(raw_config.get("target_size", 256)),
        notch_freq=float(raw_config.get("notch_freq", 60.0)),
        notch_q=float(raw_config.get("notch_q", 30.0)),
        word_window_sec=float(raw_config.get("word_window_sec", 1.0)),
        extra_window_ms_options=[int(value) for value in raw_config.get("extra_window_ms_options", [0, 100, 300])],
        preview_dpi=int(raw_config.get("preview_dpi", 256)),
        channels=channels,
        save_png=bool(raw_config.get("save_png", True)),
    )


def parse_ini_sections(text: str) -> dict[str, list[str]]:
    # BrainVision .vhdr/.vmrk files are INI-like. This helper groups raw lines by section name.
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            sections[current_section] = []
            continue
        if current_section is not None:
            sections[current_section].append(line)
    return sections


def parse_header(vhdr_path: Path) -> BrainVisionHeader:
    # BrainVision metadata lives in an INI-like text header file.
    sections = parse_ini_sections(vhdr_path.read_text(encoding="utf-8", errors="replace"))
    common = {
        key.strip(): value.strip()
        for key, value in (line.split("=", 1) for line in sections.get("Common Infos", []))
    }

    if common.get("DataFormat") != "BINARY":
        raise ValueError(f"Unsupported DataFormat in {vhdr_path}: {common.get('DataFormat')}")
    if common.get("DataOrientation") != "MULTIPLEXED":
        raise ValueError(
            f"Unsupported DataOrientation in {vhdr_path}: {common.get('DataOrientation')}"
        )

    channel_lines = sections.get("Channel Infos", [])
    channel_names: list[str] = []
    for line in channel_lines:
        _, value = line.split("=", 1)
        parts = value.split(",")
        channel_names.append(parts[0].strip())

    return BrainVisionHeader(
        data_file=common["DataFile"],
        marker_file=common["MarkerFile"],
        data_format=common["DataFormat"],
        data_orientation=common["DataOrientation"],
        number_of_channels=int(common["NumberOfChannels"]),
        sampling_interval_usec=float(common["SamplingInterval"]),
        channel_names=channel_names,
    )


def parse_markers(vmrk_path: Path) -> dict[int, int]:
    # We only keep numeric stimulus markers because they point to the audio file / trial onsets.
    sections = parse_ini_sections(vmrk_path.read_text(encoding="utf-8", errors="replace"))
    marker_lines = sections.get("Marker Infos", [])
    stimulus_positions: dict[int, int] = {}
    for line in marker_lines:
        if "=" not in line:
            continue
        _, value = line.split("=", 1)
        parts = [part.strip() for part in value.split(",")]
        if len(parts) < 3:
            continue
        marker_type, description, position = parts[:3]
        if marker_type != "Stimulus":
            continue
        stimulus_match = re.fullmatch(r"(?:S\s*)?(\d+)", description, flags=re.IGNORECASE)
        if stimulus_match is None:
            continue
        stimulus_positions[int(stimulus_match.group(1))] = int(position)
    return stimulus_positions


def is_sidecar_metadata_file(path: Path) -> bool:
    # macOS AppleDouble artifacts start with "._" and are not valid BrainVision files.
    return path.name.startswith("._")


def read_brainvision_eeg(subject_dir: Path) -> tuple[np.ndarray, BrainVisionHeader, dict[int, int]]:
    vhdr_files = sorted(
        path for path in subject_dir.glob("*.vhdr") if path.is_file() and not is_sidecar_metadata_file(path)
    )
    if not vhdr_files:
        raise FileNotFoundError(f"No .vhdr file found in {subject_dir}")
    vhdr_path = vhdr_files[0]
    header = parse_header(vhdr_path)

    eeg_path = subject_dir / header.data_file
    vmrk_path = subject_dir / header.marker_file
    raw = np.fromfile(eeg_path, dtype="<f4")
    expected_channels = header.number_of_channels
    if raw.size % expected_channels != 0:
        raise ValueError(
            f"EEG file size is not divisible by channel count for {eeg_path}: "
            f"{raw.size} values, {expected_channels} channels"
        )
    # BrainVision multiplexed binary stores samples interleaved by channel.
    eeg = raw.reshape(-1, expected_channels).T
    markers = parse_markers(vmrk_path)
    return eeg, header, markers


def select_channel_indices(channel_names: list[str], requested_channels: list[str] | None) -> list[int]:
    available_indices = [
        idx for idx, name in enumerate(channel_names) if name.upper() not in {"VEOG", "AUX5"}
    ]
    if not available_indices:
        raise ValueError("No EEG channels available after excluding VEOG/Aux5.")

    # `None` means "use all EEG channels" after removing non-EEG auxiliaries.
    if requested_channels is None:
        return available_indices

    available_names = {channel_names[idx] for idx in available_indices}
    selected_indices: list[int] = []
    seen_indices: set[int] = set()

    for token in requested_channels:
        # Channel selection accepts either 1-based numeric channel ids ("1", "2")
        # or exact channel labels from the BrainVision header ("Fz", "Cz", ...).
        if token.isdigit():
            channel_idx = int(token) - 1
            if channel_idx < 0 or channel_idx >= len(channel_names):
                raise ValueError(f"Requested channel number '{token}' is out of range.")
            channel_name = channel_names[channel_idx]
            if channel_name not in available_names:
                raise ValueError(f"Requested channel '{token}' maps to non-EEG channel '{channel_name}'.")
        else:
            if token not in available_names:
                raise ValueError(f"Requested channel name '{token}' was not found in EEG channels.")
            channel_idx = channel_names.index(token)

        if channel_idx not in seen_indices:
            selected_indices.append(channel_idx)
            seen_indices.add(channel_idx)

    if not selected_indices:
        raise ValueError("No valid EEG channels were selected.")
    return selected_indices


def apply_notch_filter(eeg: np.ndarray, sample_rate: float, notch_freq: float, q: float) -> np.ndarray:
    # Apply the notch filter once to the continuous recording before cutting word-level segments.
    b, a = signal.iirnotch(w0=notch_freq, Q=q, fs=sample_rate)
    filtered = signal.filtfilt(b, a, eeg, axis=1)
    return filtered.astype(np.float32, copy=False)
 

def compute_word_sample_bounds(
    row: pd.Series,
    marker_positions: dict[int, int],
    sample_rate: float,
    window_sec: float,
) -> tuple[int, int]:
    stimulus = int(row["stimulus"])
    if stimulus not in marker_positions:
        raise KeyError(f"Stimulus {stimulus} not found in marker file.")

    # The CSV onset is relative to the start of the stimulus audio, while the marker tells us where
    # that stimulus begins in the EEG stream. Adding them gives the word onset in EEG samples.
    marker_sample_one_based = marker_positions[stimulus]
    onset_samples = int(round(float(row["rounded_onset_sec"]) * sample_rate))
    window_samples = max(1, int(round(window_sec * sample_rate)))

    # Marker positions are 1-based in BrainVision files, so convert to 0-based before slicing numpy arrays.
    start_index = (marker_sample_one_based - 1) + onset_samples
    end_index = start_index + window_samples
    return start_index, end_index


def build_spectrogram(
    segment: np.ndarray,
    sample_rate: float,
    freq_min: float,
    freq_max: float,
    target_size: int,
    active_duration_sec: float,
    word_window_sec: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    segment_length = segment.shape[0]
    if freq_max <= freq_min:
        raise ValueError(f"freq_max must be larger than freq_min: {freq_min} !< {freq_max}")

    # Choose STFT parameters so the spectrogram lands directly on 256 time bins and 256 frequency bins
    # without resizing afterwards. Time resolution is controlled by hop_length, frequency resolution by nfft.
    # The frequency axis is now sized against the requested band width (freq_max - freq_min), not just freq_max.
    target_band_hz = freq_max - freq_min
    nfft = max(target_size + 1, int(round(sample_rate * (target_size - 1) / target_band_hz)))
    nperseg = max(8, min(segment_length, nfft))
    hop_length = max(1, math.ceil(max(segment_length - nperseg, 0) / max(target_size - 1, 1)))
    padded_length = nperseg + hop_length * (target_size - 1)
    pad_width = max(0, padded_length - segment_length)
    # Zero-padding extends the fixed 1-second segment to the exact analysis length needed for 256 frames.
    padded_segment = np.pad(segment, (0, pad_width), mode="constant")
    noverlap = nperseg - hop_length

    bin_hz = sample_rate / nfft
    start_freq_index = max(0, int(round(freq_min / bin_hz)))
    end_freq_index = start_freq_index + target_size - 1
    max_positive_freq_index = nfft // 2
    if end_freq_index > max_positive_freq_index:
        raise ValueError(
            "Computed nfft is too small to cover the requested frequency band: "
            f"start_idx={start_freq_index}, end_idx={end_freq_index}, max_idx={max_positive_freq_index}"
        )

    _, _, sxx = signal.spectrogram(
        padded_segment,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling="density",
        mode="psd",
        detrend=False,
    )

    if sxx.shape[1] != target_size:
        raise ValueError(
            f"Expected {target_size} time bins but got {sxx.shape[1]} "
            f"(segment_length={segment_length}, nperseg={nperseg}, hop={hop_length})"
        )

    # Keep exactly 256 bins starting near freq_min and ending near freq_max.
    spec = sxx[start_freq_index : end_freq_index + 1, :]
    if spec.shape[0] < target_size:
        raise ValueError(f"Expected at least {target_size} frequency bins but got {spec.shape[0]}")
    spec = spec[:target_size, :]

    active_columns = int(round(max(0.0, active_duration_sec) * target_size / word_window_sec))
    active_columns = min(target_size, max(1, active_columns))
    # Only keep the part that corresponds to the word duration (+ optional latency extension).
    if active_columns < target_size:
        spec[:, active_columns:] = 0.0

    actual_freq_min = start_freq_index * bin_hz
    actual_freq_max = end_freq_index * bin_hz

    # These parameters are written to metadata so downstream users can reconstruct the STFT setup.
    params = {
        "nfft": int(nfft),
        "nperseg": int(nperseg),
        "hop_length": int(hop_length),
        "noverlap": int(noverlap),
        "padded_length": int(padded_length),
        "pad_width": int(pad_width),
        "active_time_bins": int(active_columns),
        "start_freq_index": int(start_freq_index),
        "end_freq_index": int(end_freq_index),
        "freq_bin_hz": float(bin_hz),
        "actual_freq_min_hz": float(actual_freq_min),
        "actual_freq_max_hz": float(actual_freq_max),
    }
    return spec.astype(np.float32), params


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_png(
    array_2d: np.ndarray,
    raw_trace: np.ndarray,
    output_path: Path,
    preview_dpi: int,
    sample_rate: float,
    freq_min_hz: float,
    freq_max_hz: float,
    figure_title: str,
) -> None:
    # Use a writable temporary matplotlib cache so image export works inside sandboxed environments too.
    cache_dir = Path(tempfile.gettempdir()) / "eeg_spectrogram_matplotlib"
    ensure_dir(cache_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # PNGs are quick previews; the authoritative data stays in .npy files.
    duration_sec = raw_trace.shape[0] / sample_rate
    time_axis = np.arange(raw_trace.shape[0], dtype=np.float32) / sample_rate
    time_start_sec = 0.0
    time_end_sec = duration_sec

    fig = plt.figure(figsize=(6.5, 5), dpi=preview_dpi, constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[3, 1],
        width_ratios=[40, 2],
    )
    spec_ax = fig.add_subplot(grid[0, 0])
    trace_ax = fig.add_subplot(grid[1, 0], sharex=spec_ax)
    colorbar_ax = fig.add_subplot(grid[0, 1])

    image = spec_ax.imshow(
        array_2d,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[time_start_sec, time_end_sec, freq_min_hz, freq_max_hz],
    )
    spec_ax.set_ylabel("Frequency (Hz)")
    spec_ax.set_title(figure_title)
    fig.colorbar(image, cax=colorbar_ax, label="Amplitude")

    trace_ax.plot(time_axis, raw_trace, color="black", linewidth=0.8)
    trace_ax.set_xlabel("Time (s)")
    trace_ax.set_ylabel("EEG")
    spec_ax.set_xlim(time_start_sec, time_end_sec)
    trace_ax.set_xlim(time_start_sec, time_end_sec)
    tick_positions = np.linspace(time_start_sec, time_end_sec, num=6)
    spec_ax.set_xticks(tick_positions)
    trace_ax.set_xticks(tick_positions)

    fig.savefig(output_path, dpi=preview_dpi, bbox_inches="tight")
    plt.close(fig)


def subject_dirs(data_root: Path, subjects: Iterable[str] | None) -> list[Path]:
    # If no subject list is provided, scan the raw EEG root and process every Sxx directory.
    if subjects:
        return [data_root / subject for subject in subjects]
    return sorted(path for path in data_root.iterdir() if path.is_dir() and path.name.startswith("S"))


def sanitize_token(value: object) -> str:
    # Keep output paths stable and filesystem-safe even when words contain punctuation or spaces.
    text = str(value)
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^A-Za-z0-9_.-]+", "", text)
    return text or "unknown"


def normalize_word_token(value: object) -> str:
    return str(value).strip().lower()


def build_output_stem(
    subject_name: str,
    stimulus: int,
    interval_index: int,
    word: object,
    extra_window_ms: int,
) -> str:
    # Keep filenames aligned with the requested dataset export convention.
    return (
        f"{subject_name}_DownTheRabbitHoleFinal_"
        f"SoundFIle{stimulus}_{interval_index:04d}_{sanitize_token(word)}_{extra_window_ms}ms"
    )


def process_subject(
    subject_dir: Path,
    timing_df: pd.DataFrame,
    config: AppConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    eeg, header, markers = read_brainvision_eeg(subject_dir)
    # BrainVision stores sampling interval in microseconds, so convert it back to Hz.
    header_sample_rate = 1_000_000.0 / header.sampling_interval_usec
    effective_sample_rate = (
        float(config.sample_rate) if config.sample_rate is not None else header_sample_rate
    )
    channel_indices = select_channel_indices(header.channel_names, config.channels)
    filtered = apply_notch_filter(
        eeg[channel_indices], effective_sample_rate, config.notch_freq, config.notch_q
    )

    subject_output_dir = config.output_root / subject_dir.name
    ensure_dir(subject_output_dir)

    metadata_rows: list[dict[str, object]] = []
    # Normalization is done after all log spectrograms are measured so each latency condition
    # can use the subject-wide max computed from its own 0/100/300 ms subset.
    normalization_queue: list[dict[str, object]] = []
    subject_log_raw_max_by_latency: dict[int, float] = defaultdict(float)
    total_samples = filtered.shape[1]
    total_variants = len(timing_df) * len(config.extra_window_ms_options)
    progress_bar = TerminalProgressBar(total_variants, f"{subject_dir.name} variants")

    for row in timing_df.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        start_sample, end_sample = compute_word_sample_bounds(
            row_series, markers, effective_sample_rate, config.word_window_sec
        )
        if end_sample > total_samples:
            raise ValueError(
                f"{subject_dir.name} segment exceeds EEG length for stimulus {row_series['stimulus']} "
                f"interval {row_series['interval_index']}: {end_sample} > {total_samples}"
            )

        word_dir = subject_output_dir / sanitize_token(row_series["word"])
        ensure_dir(word_dir)
        # This segment is always the fixed analysis window (default 1 second) starting at the word onset.
        segment = filtered[:, start_sample:end_sample]
        preview_channel_name = header.channel_names[channel_indices[0]]
        preview_channel_number = channel_indices[0] + 1
        base_duration_sec = float(row_series["rounded_duration_sec"])

        # Save one set of outputs for each latency condition: 0 ms, +100 ms, +300 ms, ...
        for extra_window_ms in config.extra_window_ms_options:
            # Folder layout: <word>/<latency_ms>/log_normalized
            variant_dir = word_dir / str(extra_window_ms)
            log_normalized_dir = variant_dir / "log_normalized"
            ensure_dir(log_normalized_dir)

            # The underlying EEG window stays fixed, but the visible non-zero part grows by the
            # requested latency extension and is clipped at the 1-second analysis window.
            active_duration_sec = min(
                config.word_window_sec,
                base_duration_sec + (extra_window_ms / 1000.0),
            )
            variant_stem = build_output_stem(
                subject_name=subject_dir.name,
                stimulus=int(row_series["stimulus"]),
                interval_index=int(row_series["interval_index"]),
                word=row_series["word"],
                extra_window_ms=int(extra_window_ms),
            )

            channel_spectrograms: list[np.ndarray] = []
            spec_params: dict[str, float | int] | None = None
            for segment_channel_idx, header_channel_idx in enumerate(channel_indices):
                spectrogram, current_spec_params = build_spectrogram(
                    segment=segment[segment_channel_idx],
                    sample_rate=effective_sample_rate,
                    freq_min=config.freq_min,
                    freq_max=config.freq_max,
                    target_size=config.target_size,
                    active_duration_sec=active_duration_sec,
                    word_window_sec=config.word_window_sec,
                )
                channel_spectrograms.append(spectrogram)
                if spec_params is None:
                    spec_params = current_spec_params

            if spec_params is None:
                raise ValueError("No channel spectrograms were generated.")

            # Stored array shape is always channels x 256 x 256.
            raw_stacked_spectrogram = np.stack(channel_spectrograms, axis=0)
            log_stacked_spectrogram = np.log10(np.maximum(raw_stacked_spectrogram, 0.0) + 1e-12).astype(np.float32)
            current_log_raw_max = float(np.max(log_stacked_spectrogram))
            subject_log_raw_max_by_latency[extra_window_ms] = max(
                subject_log_raw_max_by_latency[extra_window_ms], current_log_raw_max
            )

            # Metadata is one row per (word, latency condition), not one row per channel.
            metadata_row = {
                "subject": subject_dir.name,
                "stimulus": int(row_series["stimulus"]),
                "interval_index": int(row_series["interval_index"]),
                "word": row_series["word"],
                "extra_window_ms": int(extra_window_ms),
                "selected_channel_names": ",".join(header.channel_names[idx] for idx in channel_indices),
                "num_channels": int(len(channel_indices)),
                "preview_channel_name": preview_channel_name,
                "preview_channel_number": int(preview_channel_number),
                "rounded_onset_sec": float(row_series["rounded_onset_sec"]),
                "rounded_duration_sec": base_duration_sec,
                "effective_duration_sec": active_duration_sec,
                "marker_sample_one_based": int(markers[int(row_series["stimulus"])]),
                "segment_start_sample_zero_based": start_sample,
                "segment_end_sample_zero_based": end_sample,
                "segment_num_samples": int(segment.shape[1]),
                "header_sample_rate_hz": float(header_sample_rate),
                "effective_sample_rate_hz": float(effective_sample_rate),
                "nfft": spec_params["nfft"],
                "nperseg": spec_params["nperseg"],
                "hop_length": spec_params["hop_length"],
                "noverlap": spec_params["noverlap"],
                "padded_length": spec_params["padded_length"],
                "pad_width": spec_params["pad_width"],
                "active_time_bins": spec_params["active_time_bins"],
                "start_freq_index": spec_params["start_freq_index"],
                "end_freq_index": spec_params["end_freq_index"],
                "freq_bin_hz": spec_params["freq_bin_hz"],
                "actual_freq_min_hz": spec_params["actual_freq_min_hz"],
                "actual_freq_max_hz": spec_params["actual_freq_max_hz"],
                "spectrogram_shape": "x".join(str(dim) for dim in raw_stacked_spectrogram.shape),
                "latency_log_raw_max_amplitude": "",
                "subject_log_raw_max_amplitude": "",
                "spectrogram_npy_log_normalized": "",
                "spectrogram_png_log_normalized": "",
            }
            metadata_rows.append(metadata_row)
            normalization_queue.append(
                {
                    "metadata_row": metadata_row,
                    "log_normalized_dir": log_normalized_dir,
                    "variant_stem": variant_stem,
                    "preview_channel_name": preview_channel_name,
                    "preview_channel_number": int(preview_channel_number),
                    "preview_trace": segment[0].astype(np.float32, copy=True) if config.save_png else None,
                    "log_array": log_stacked_spectrogram,
                    "extra_window_ms": extra_window_ms,
                }
            )
            progress_bar.update(
                suffix=(
                    f"word={row_series['word']} "
                    f"stim={int(row_series['stimulus'])} "
                    f"idx={int(row_series['interval_index'])} "
                    f"+{int(extra_window_ms)}ms"
                )
            )

    for item in normalization_queue:
        metadata_row = item["metadata_row"]
        extra_window_ms = int(item["extra_window_ms"])
        log_normalization_denominator = (
            subject_log_raw_max_by_latency[extra_window_ms]
            if subject_log_raw_max_by_latency[extra_window_ms] > 0.0
            else 1.0
        )
        log_array = np.asarray(item["log_array"], dtype=np.float32)
        log_normalized_array = (log_array / log_normalization_denominator).astype(np.float32)
        log_normalized_npy_path = item["log_normalized_dir"] / f"{item['variant_stem']}_log_norm.npy"
        np.save(log_normalized_npy_path, log_normalized_array)

        log_normalized_png_path = ""
        if config.save_png:
            preview_trace = np.asarray(item["preview_trace"], dtype=np.float32)
            figure_title = (
                f"word={metadata_row['word']} | stimulus={int(metadata_row['stimulus'])} | "
                f"interval_index={int(metadata_row['interval_index'])} | "
                f"extra_window_ms={int(metadata_row['extra_window_ms'])} | "
                f"channel={int(item['preview_channel_number'])}"
            )
            log_normalized_png_file = (
                item["log_normalized_dir"]
                / f"{item['variant_stem']}_log_norm_preview_ch_{sanitize_token(item['preview_channel_name'])}.png"
            )
            save_png(
                log_normalized_array[0],
                preview_trace,
                log_normalized_png_file,
                config.preview_dpi,
                effective_sample_rate,
                float(metadata_row["actual_freq_min_hz"]),
                float(metadata_row["actual_freq_max_hz"]),
                figure_title,
            )
            log_normalized_png_path = str(log_normalized_png_file)

        metadata_row["latency_log_raw_max_amplitude"] = float(
            subject_log_raw_max_by_latency[extra_window_ms]
        )
        metadata_row["subject_log_raw_max_amplitude"] = float(
            max(subject_log_raw_max_by_latency.values(), default=0.0)
        )
        metadata_row["spectrogram_npy_log_normalized"] = str(log_normalized_npy_path)
        metadata_row["spectrogram_png_log_normalized"] = log_normalized_png_path

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = subject_output_dir / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False, quoting=csv.QUOTE_MINIMAL)
    # Summary keeps one row per subject and records the normalization maxima used for each latency.
    summary_info = {
        "subject": subject_dir.name,
        "num_spectrograms": len(metadata_df),
        "output_dir": str(subject_output_dir),
        "subject_log_raw_max_amplitude": float(
            max(subject_log_raw_max_by_latency.values(), default=0.0)
        ),
        "log_raw_max_amplitude_0ms": float(subject_log_raw_max_by_latency.get(0, 0.0)),
        "log_raw_max_amplitude_100ms": float(subject_log_raw_max_by_latency.get(100, 0.0)),
        "log_raw_max_amplitude_300ms": float(subject_log_raw_max_by_latency.get(300, 0.0)),
    }
    return metadata_df, summary_info


def main() -> None:
    config = load_config()
    # The timing CSV defines which words exist, when they start relative to each stimulus,
    # and how long each word lasts.
    timing_df = pd.read_csv(config.timing_csv)
    required_columns = {"stimulus", "interval_index", "word", "rounded_onset_sec", "rounded_duration_sec"}
    missing = required_columns.difference(timing_df.columns)
    if missing:
        raise ValueError(f"Missing required timing CSV columns: {sorted(missing)}")

    if config.words is not None:
        requested_words = {normalize_word_token(word) for word in config.words}
        available_words = timing_df["word"].astype(str).map(normalize_word_token)
        matched_words = set(available_words[available_words.isin(requested_words)].unique().tolist())
        missing_words = sorted(requested_words.difference(matched_words))
        if missing_words:
            raise ValueError(
                "Some requested words were not found in the timing CSV: "
                f"{missing_words}. Available words include: "
                f"{sorted(timing_df['word'].astype(str).unique().tolist())}"
            )
        timing_df = timing_df[available_words.isin(requested_words)].copy()
        if timing_df.empty:
            raise ValueError(f"No rows matched requested words: {sorted(requested_words)}")

    ensure_dir(config.output_root)
    subjects = subject_dirs(config.data_root, config.subjects)
    if not subjects:
        raise FileNotFoundError(f"No subject directories found under {config.data_root}")

    summary_rows: list[dict[str, object]] = []
    subject_progress_bar = TerminalProgressBar(len(subjects), "subjects")
    for subject_dir in subjects:
        metadata_df, summary_info = process_subject(subject_dir, timing_df, config)
        summary_rows.append(summary_info)
        subject_progress_bar.update(suffix=f"{subject_dir.name} complete")
        print(
            f"[OK] {subject_dir.name}: generated {len(metadata_df)} spectrograms in "
            f"{config.output_root / subject_dir.name}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(config.output_root / "summary.csv", index=False)


if __name__ == "__main__":
    main()
