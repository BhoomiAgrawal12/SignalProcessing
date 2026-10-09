"""Constellation definitions with Gray bit mappings, shared by the
modulator (synth factory) and the demodulator/slicer."""
from __future__ import annotations

import numpy as np


def _psk(order: int) -> tuple:
    """Gray-coded PSK. Returns (points, bits_per_symbol, labels)."""
    k = int(np.log2(order))
    gray = np.arange(order) ^ (np.arange(order) >> 1)
    pts = np.exp(1j * (2 * np.pi * np.arange(order) / order +
                       (np.pi / 4 if order == 4 else 0)))
    labels = np.zeros((order, k), dtype=np.uint8)
    points = np.zeros(order, dtype=np.complex128)
    for idx in range(order):
        g = gray[idx]
        points[g] = pts[idx] if False else pts[idx]
        # store mapping label->point: label g at angle position idx
    # build label -> point table directly
    table = np.zeros(order, dtype=np.complex128)
    for pos in range(order):
        table[gray[pos]] = pts[pos]
    for lab in range(order):
        labels[lab] = [(lab >> (k - 1 - j)) & 1 for j in range(k)]
    return table, k


def _qam(order: int) -> tuple:
    side = int(np.sqrt(order))
    k = int(np.log2(order))
    kb = k // 2
    gray = np.arange(side) ^ (np.arange(side) >> 1)
    # gray[i] is the gray code of index i; we need position of each label
    pos_of = np.argsort(gray)
    levels = 2 * np.arange(side) - (side - 1)
    table = np.zeros(order, dtype=np.complex128)
    for lab in range(order):
        li = (lab >> kb) & ((1 << kb) - 1)
        lq = lab & ((1 << kb) - 1)
        table[lab] = levels[pos_of[li]] + 1j * levels[pos_of[lq]]
    table /= np.sqrt((np.abs(table) ** 2).mean())
    return table, k


def _ask(order: int) -> tuple:
    """Bipolar amplitude-shift keying (PAM levels on the real axis).
    OOK is the order-2 special case; after S1 conditioning removes the DC
    term an on/off signal lands on symmetric bipolar levels anyway."""
    k = int(np.log2(order))
    gray = np.arange(order) ^ (np.arange(order) >> 1)
    pos_of = np.argsort(gray)
    levels = 2 * np.arange(order) - (order - 1)
    table = np.zeros(order, dtype=np.complex128)
    for lab in range(order):
        table[lab] = levels[pos_of[lab]]
    table /= np.sqrt((np.abs(table) ** 2).mean())
    return table, k


def _cross_qam(order: int) -> tuple:
    """Cross constellations for 32/128 QAM: square grid with the four
    corner blocks removed (standard cross shape)."""
    k = int(np.log2(order))
    side = {32: 6, 128: 12}[order]
    cut = {32: 1, 128: 2}[order]
    pts = []
    for i in range(side):
        for q in range(side):
            if (i < cut or i >= side - cut) and (q < cut or q >= side - cut):
                continue
            pts.append(complex(2 * i - (side - 1), 2 * q - (side - 1)))
    pts = np.array(pts[:order])
    pts /= np.sqrt((np.abs(pts) ** 2).mean())
    # nearest-neighbour ordered labelling (consistent, shared with the
    # modulator; blind analysis never depends on a standard mapping)
    order_idx = np.lexsort((pts.imag, pts.real))
    table = pts[order_idx]
    return table, k


def _apsk(order: int) -> tuple:
    """Ring constellations in the DVB-S2(X) style: (points per ring,
    relative radius, phase offset). Labels are assigned ring-major, which
    the factory shares, so encode/decode round-trips exactly."""
    rings = {
        16: [(4, 1.0, np.pi / 4), (12, 2.57, 0.0)],
        32: [(4, 1.0, np.pi / 4), (12, 2.53, 0.0), (16, 4.30, np.pi / 16)],
        64: [(8, 1.0, np.pi / 8), (16, 2.2, 0.0), (20, 3.6, np.pi / 20),
             (20, 5.2, 0.0)],
        128: [(8, 1.0, np.pi / 8), (16, 2.0, 0.0), (24, 3.2, np.pi / 24),
              (32, 4.6, 0.0), (48, 6.2, np.pi / 48)],
    }[order]
    pts = []
    for n_pts, radius, phase in rings:
        ang = 2 * np.pi * np.arange(n_pts) / n_pts + phase
        pts.extend(radius * np.exp(1j * ang))
    pts = np.array(pts[:order])
    pts /= np.sqrt((np.abs(pts) ** 2).mean())
    return pts, int(np.log2(order))


CONSTELLATIONS = {
    "BPSK": _psk(2),
    "QPSK": _psk(4),
    "OQPSK": _psk(4),          # offset applied in the waveform, not the map
    "8PSK": _psk(8),
    "16PSK": _psk(16),
    "32PSK": _psk(32),
    "OOK": _ask(2),
    "4ASK": _ask(4),
    "8ASK": _ask(8),
    "16QAM": _qam(16),
    "32QAM": _cross_qam(32),
    "64QAM": _qam(64),
    "128QAM": _cross_qam(128),
    "256QAM": _qam(256),
    "16APSK": _apsk(16),
    "32APSK": _apsk(32),
    "64APSK": _apsk(64),
    "128APSK": _apsk(128),
}

def constellation_symmetry(name: str) -> int:
    """Order of the rotational symmetry group: the largest k such that
    rotation by 2pi/k maps the constellation onto itself. This is the
    number of indistinguishable carrier-lock rotations a blind receiver
    must enumerate (e.g. 4 for square QAM, 8 for these 128APSK rings,
    M for M-PSK)."""
    table, _k = CONSTELLATIONS[name]
    pts = np.sort_complex(np.round(table, 6))
    best = 1
    for k in (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64):
        rot = np.sort_complex(np.round(table * np.exp(2j * np.pi / k), 6))
        if np.allclose(np.abs(pts - rot), 0, atol=1e-4):
            best = k
    return best


_SYMMETRY_CACHE = {name: None for name in CONSTELLATIONS}


def symmetry_order(name: str) -> int:
    if _SYMMETRY_CACHE.get(name) is None:
        _SYMMETRY_CACHE[name] = constellation_symmetry(name)
    return _SYMMETRY_CACHE[name]


# family lookup used by the receiver dispatch and the quality gate
MOD_FAMILY = {}
for _name in CONSTELLATIONS:
    if _name == "OQPSK":
        MOD_FAMILY[_name] = "oqpsk"
    elif _name.endswith("APSK"):
        MOD_FAMILY[_name] = "apsk"
    elif _name.endswith("PSK"):
        MOD_FAMILY[_name] = "psk"
    elif _name.endswith("QAM"):
        MOD_FAMILY[_name] = "qam"
    else:
        MOD_FAMILY[_name] = "ask"
for _name in ("2FSK", "4FSK", "8FSK"):
    MOD_FAMILY[_name] = "fsk"
MOD_FAMILY["GMSK"] = "gmsk"
for _name in ("FM", "AM-DSB-WC", "AM-DSB-SC", "AM-SSB-WC", "AM-SSB-SC"):
    MOD_FAMILY[_name] = "analog"


def bits_to_iq_symbols(bits: np.ndarray, modulation: str) -> np.ndarray:
    table, k = CONSTELLATIONS[modulation]
    n_sym = len(bits) // k
    b = np.asarray(bits[: n_sym * k], dtype=np.int64).reshape(n_sym, k)
    weights = 1 << np.arange(k - 1, -1, -1)
    labels = (b * weights).sum(axis=1)
    return table[labels]


def slice_symbols(symbols: np.ndarray, modulation: str,
                  noise_var: float = 0.1) -> tuple:
    """Nearest-point slicing + max-log LLRs.

    Returns (hard_bits, llrs, evm_percent). LLR > 0 means bit 0 more
    likely (convention used throughout the bit layer)."""
    table, k = CONSTELLATIONS[modulation]
    d2 = np.abs(symbols[:, None] - table[None, :]) ** 2   # (N, order)
    nearest = np.argmin(d2, axis=1)
    err = symbols - table[nearest]
    evm = float(np.sqrt((np.abs(err) ** 2).mean() /
                        (np.abs(table) ** 2).mean()) * 100)
    order = len(table)
    labels = np.arange(order)
    hard = np.zeros((len(symbols), k), dtype=np.uint8)
    llrs = np.zeros((len(symbols), k), dtype=np.float32)
    nv = max(noise_var, 1e-6)
    for j in range(k):
        bit_of_label = (labels >> (k - 1 - j)) & 1
        d0 = d2[:, bit_of_label == 0].min(axis=1)
        d1 = d2[:, bit_of_label == 1].min(axis=1)
        llrs[:, j] = (d1 - d0) / nv
        hard[:, j] = (llrs[:, j] < 0)
    return hard.ravel(), llrs.ravel(), evm
