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


def load_audio(path: str):
  waveform, _ = librosa.load(path, sr=SAMPLE_RATE)
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


def process_audio_waveform(waveform: np.ndarray, segment_time_stamps: list) -> list:
  sample_slice_indices = [int(t * SAMPLE_RATE) for t in segment_time_stamps]
  sample_slice_indices.insert(0, 0)
  segments = []
  for i in range(0,len(segment_time_stamps)):
    segments.append(waveform[sample_slice_indices[i]:sample_slice_indices[i] + (30 * SAMPLE_RATE)])
  segments = [np.pad(seg, (0, SAMPLES_PER_SEGMENT - len(seg)), 'constant') for seg in segments]


  window = torch.hann_window(N_FFT)
  filters = torch.from_numpy(librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)).float()

  log_specs = []
  for segment in segments:
    waveform_tensor = torch.from_numpy(segment).float()
    stft = torch.stft(waveform_tensor, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    mel_spec = filters @ magnitudes
    # Convert to log scale safely
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    log_specs.append(log_spec)

  return log_specs
