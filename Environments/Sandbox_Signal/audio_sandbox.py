# Environments/Sandbox_Signal/audio_sandbox.py
import torch
import torchaudio
import math
import random


class AudioSandbox:
    def __init__(self, sample_rate=8000, duration=0.5):
        self.sr = sample_rate
        self.duration = duration
        self.num_samples = int(self.sr * self.duration)

        # Bộ chuyển đổi Sóng âm -> Biểu đồ phổ tần số (Mel Spectrogram)
        # Giới hạn 20 dải tần (20 input neurons cho tai)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=400,
            hop_length=200,
            n_mels=20,
        )

    def generate_sound(self, sound_type):
        """Sinh sóng âm và chuyển thành tín hiệu đầu vào cho SNN"""
        t = torch.linspace(0, self.duration, self.num_samples)

        if sound_type == 0:
            # ID 0: Tĩnh lặng (Chỉ có tiếng xì xèo của nhiễu nền)
            waveform = torch.randn(self.num_samples) * 0.05

        elif sound_type == 1:
            # ID 1: Tiếng Trầm (Giả lập tiếng người nói - Tần số 300Hz)
            waveform = torch.sin(2 * math.pi * 300 * t) + (torch.randn(self.num_samples) * 0.05)

        elif sound_type == 2:
            # ID 2: Âm Thanh Lạ (Giả lập tiếng chim hót/Còi báo động - Tần số cao 2000Hz)
            waveform = torch.sin(2 * math.pi * 2000 * t) + (torch.randn(self.num_samples) * 0.05)

        # Chuyển sóng âm thành Phổ tần số (Tần số x Thời gian)
        # Shape: [1, 20 mels, time_steps]
        mel_spec = self.mel_spectrogram(waveform.unsqueeze(0))

        # SNN cần đầu vào theo chuẩn Thời gian: [time_steps, batch=1, input_features=20]
        mel_spec = mel_spec.squeeze(0).transpose(0, 1).unsqueeze(1)

        # Chuẩn hóa tín hiệu (Đưa về dải 0.0 -> 1.0) để nơ-ron không bị sốc điện
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)
        return mel_spec