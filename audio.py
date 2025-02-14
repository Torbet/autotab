import numpy as np
import torch
import torch.nn.functional as F
import librosa

SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
SAMPLES_PER_SEGMENT = SAMPLE_RATE * 30
FRAMES_PER_SEGMENT = SAMPLES_PER_SEGMENT // HOP_LENGTH


def load_audio(waveform: np.ndarray):
  waveform = (
    waveform[..., :SAMPLES_PER_SEGMENT]
    if waveform.shape[-1] > SAMPLES_PER_SEGMENT
    else np.pad(waveform, (0, SAMPLES_PER_SEGMENT - waveform.shape[-1]))
  )
  waveform = torch.from_numpy(waveform).float()
  window = torch.hann_window(N_FFT)
  stft = torch.stft(waveform, N_FFT, HOP_LENGTH, window=window, return_complex=True)
  magnitudes = stft[..., :-1].abs() ** 2
  filters = torch.from_numpy(librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)).float()
  mel_spec = filters @ magnitudes
  log_spec = torch.clamp(mel_spec, min=1e-10).log10()
  log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
  log_spec = (log_spec + 4.0) / 4.0
  return log_spec


def process_audio_waveform(waveform: np.ndarray, segment_timestamps: list) -> list:
  slice_indices = [(int(s * SAMPLE_RATE), int(f * SAMPLE_RATE)) for s, f in segment_timestamps]

  log_specs = []
  for s, f in slice_indices:
    segment = waveform[s:f]
    log_specs.append(load_audio(segment))

  return log_specs
