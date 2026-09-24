# PyTorch Compatibility Matrix

> Sources: [pytorch.org/get-started/previous-versions](https://pytorch.org/get-started/previous-versions/) and [github.com/pytorch/torchcodec](https://github.com/pytorch/torchcodec#installing-torchcodec)

---

## Notes

- **PyTorch 2.8.1 does not exist** — the release was `2.8.0`. It is listed below as such.
- torchcodec is a separate install and must match the torch version (see install commands below).
- torchaudio 2.8+ delegates audio I/O to torchcodec under the hood; from 2.9 onward it is fully in maintenance mode.

---

## Compatibility Table

| PyTorch Version | CUDA | torchvision | torchaudio | torchcodec |
|-----------------|------|-------------|------------|------------|
| 2.8.0 | cu128 | 0.23.0 | 2.8.0 | 0.7.x |
| 2.9.0 | cu128 | 0.24.0 | 2.9.0 | 0.8.x / 0.9.x |
| 2.9.0 | cu130 | 0.24.0 | 2.9.0 | 0.8.x / 0.9.x |
| 2.10.0 | cu130 | 0.25.0 | 2.10.0 | 0.10.x |

---

## pip Install Commands

### PyTorch 2.8.0 + CUDA 12.8
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install torchcodec==0.7.0 --index-url https://download.pytorch.org/whl/cu128
```

### PyTorch 2.9.0 + CUDA 12.8
```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install torchcodec==0.9.0 --index-url https://download.pytorch.org/whl/cu128
```

### PyTorch 2.9.0 + CUDA 13.0
```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130
pip install torchcodec==0.9.0 --index-url https://download.pytorch.org/whl/cu130
```

### PyTorch 2.10.0 + CUDA 13.0
```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
pip install torchcodec==0.10.0 --index-url https://download.pytorch.org/whl/cu130
```

---

## torchcodec Version Map (official)

| torchcodec | torch |
|------------|-------|
| 0.7 | 2.8 |
| 0.8 | 2.9 |
| 0.9 | 2.9 |
| 0.10 | 2.10 |
| 0.11 | 2.11 |

> ⚠️ torchcodec 0.8 and 0.9 are both listed as compatible with torch 2.9 in the official docs. Use 0.9 as it is the newer stable release for that torch version.
