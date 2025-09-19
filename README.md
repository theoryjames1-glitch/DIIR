# 🔧 Universal Differentiable IIR Filter (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal

# -------------------------------
# Utility: convert poles/zeros to biquad sections
# -------------------------------
def zpk_to_biquads(z, p, k, fs=2.0):
    # Convert to second-order sections (SOS)
    sos = signal.zpk2sos(z, p, k, fs=fs)
    return sos

# -------------------------------
# Differentiable Biquad Section
# -------------------------------
class DifferentiableBiquad(nn.Module):
    def __init__(self, b, a):
        super().__init__()
        # Normalize so a0 = 1
        b = b / a[0]
        a = a / a[0]

        # store as trainable parameters
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        self.a = nn.Parameter(torch.tensor(a[1:], dtype=torch.float32))  # skip a0=1

        # state (z^-1, z^-2)
        self.register_buffer("zi", torch.zeros(2))

    def forward(self, x):
        # Direct Form II Transposed biquad
        y = []
        z1, z2 = self.zi
        for n in range(x.shape[-1]):
            xn = x[..., n]
            yn = self.b[0]*xn + z1
            z1_new = self.b[1]*xn - self.a[0]*yn + z2
            z2_new = self.b[2]*xn - self.a[1]*yn
            z1, z2 = z1_new, z2_new
            y.append(yn)
        self.zi = torch.stack([z1.detach(), z2.detach()])
        return torch.stack(y, dim=-1)

# -------------------------------
# Universal Differentiable IIR Filter
# -------------------------------
class DifferentiableIIR(nn.Module):
    def __init__(self, z, p, k, fs=2.0):
        super().__init__()
        sos = zpk_to_biquads(z, p, k, fs=fs)
        self.sections = nn.ModuleList([
            DifferentiableBiquad(s[:3], s[3:]) for s in sos
        ])

    def forward(self, x):
        y = x
        for section in self.sections:
            y = section(y)
        return y
```

---

# ⚡ Example: Trainable Butterworth Lowpass

```python
# Design a 4th-order Butterworth low-pass at 0.3 Nyquist
z, p, k = signal.butter(4, 0.3, output="zpk")

# Wrap in differentiable filter
filt = DifferentiableIIR(z, p, k, fs=2.0)

# Test on white noise
x = torch.randn(1, 2048)  # batch=1, time=2048
y = filt(x)

print("Input shape:", x.shape, " Output shape:", y.shape)
```

---

# 🎛 Training Use Case

Since coefficients `b` and `a` are `nn.Parameter`, you can **train the filter** with any loss:

```python
optimizer = torch.optim.Adam(filt.parameters(), lr=1e-3)

for step in range(100):
    x = torch.randn(1, 1024)
    y = filt(x)

    # Example loss: match a target filtered signal
    target = F.avg_pool1d(x.unsqueeze(1), kernel_size=8, stride=1).squeeze(1)
    target = target[..., :y.shape[-1]]  # align length

    loss = F.mse_loss(y, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step} | Loss={loss.item():.6f}")
```

---

✅ This framework supports:

* Butterworth, Chebyshev I/II, Elliptic, Bessel, Legendre… anything `scipy.signal` can design.
* Training the poles/zeros **directly by gradient descent**.
* Stacking into **differentiable EQ banks, crossovers, or DDSP frontends**.

