"""Generate the fixtures for web/test-node.js using the synthetic factory."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import scipy.io.wavfile as wavfile

from dhwani.synth.factory import WaveformFactory

out = os.path.join(os.path.dirname(__file__), "..", "examples", "webtest")
os.makedirs(out, exist_ok=True)
fac = WaveformFactory(seed=42)

iq, _ = fac.generate(modulation="QPSK", sps=8.0, snr_db=20.0, cfo_norm=0.006,
                     phase_offset=0.4, n_frames=120)
iq.astype(np.complex64).tofile(os.path.join(out, "qpsk_c64.iq"))

iq2, _ = fac.generate(modulation="2FSK", sps=8.0, snr_db=16.0, n_frames=120)
x = iq2 / np.abs(iq2).max() * 0.7
inter = np.empty(2 * len(x), np.int16)
inter[0::2] = (x.real * 32000).astype(np.int16)
inter[1::2] = (x.imag * 32000).astype(np.int16)
inter.tofile(os.path.join(out, "fsk2_int16.iq"))

iq3, _ = fac.generate(modulation="16QAM", sps=8.0, snr_db=25.0,
                      cfo_norm=0.003, n_frames=120)
y = iq3 / np.abs(iq3).max() * 0.8
stereo = np.stack([(y.real * 30000).astype(np.int16),
                   (y.imag * 30000).astype(np.int16)], axis=1)
wavfile.write(os.path.join(out, "qam16.wav"), 96000, stereo)

iq4, _ = fac.generate(modulation="BPSK", sps=8.0, snr_db=15.0, n_frames=120)
z = iq4 / np.abs(iq4).max() * 0.6
u8 = np.empty(2 * len(z), np.uint8)
u8[0::2] = (z.real * 100 + 127.5).astype(np.uint8)
u8[1::2] = (z.imag * 100 + 127.5).astype(np.uint8)
u8.tofile(os.path.join(out, "bpsk_u8.iq"))
# mono (real) WAV: a 2FSK burst on a 0.15-cycle/sample carrier; both
# engines convert it to complex baseband at half the rate (1.A), so 16
# samples/symbol here arrive as 8
iq5, _ = fac.generate(modulation="2FSK", sps=16.0, snr_db=16.0, n_frames=120)
n = np.arange(len(iq5))
real = (iq5 * np.exp(2j * np.pi * 0.15 * n)).real
real = real / np.abs(real).max() * 0.7
wavfile.write(os.path.join(out, "fsk2_mono.wav"), 48000,
              (real * 32000).astype(np.int16))
# 2FSK at 4 samples/symbol (window-ranking fix, both engines)
iq6, _ = fac.generate(modulation="2FSK", sps=4.0, snr_db=16.0, n_frames=120)
iq6.astype(np.complex64).tofile(os.path.join(out, "fsk2_4sps.iq"))
print("fixtures written to", out)
