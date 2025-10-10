# SPDX-License-Identifier: Apache-2.0
import sys
import time
import random
from pathlib import Path

import torch
import ttnn

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from examples.composable.sqrt.host import host as sqrt_host

REF_FN = ttnn.sqrt
MAKO_HOST_AND_KERNEL = sqrt_host 

def main():
    torch.manual_seed(42)
    shape = (random.randint(1, 64), random.randint(1, 64))
    # For sqrt, we need positive values to avoid NaN
    a = torch.rand(shape, dtype=torch.bfloat16) + 0.1  # ∈ [0.1, 1.1)
    b = torch.rand(shape, dtype=torch.bfloat16) + 0.1  # ∈ [0.1, 1.1)

    device = ttnn.open_device(device_id=0)

    try:

        print(f"shape: {shape}")
        print(f"A range: [{a.min().item():.4f}, {a.max().item():.4f}]")
        print("Comparison to torch.sqrt")
        print()

        a_tt = ttnn.from_torch(a, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        b_tt = ttnn.from_torch(b, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # Get torch reference
        torch_ref = torch.sqrt(a.float()).to(torch.bfloat16)
        
        t0 = time.time()
        ttnn_tt = REF_FN(a_tt)
        ttnn_result = ttnn.to_torch(ttnn_tt, device=device)
        ttnn_ms = (time.time() - t0) * 1000.0

        print("-" * 60)
        print(f"out shape: {tuple(ttnn_result.shape)}")
        print(f"{'Implementation':<22} {'Allclose':<10} {'Max Error':<14} {'Avg Error':<14} {'Time (ms)':<10}")
        print("-" * 60)
        
        # Compare TTNN vs Torch
        ttnn_vs_torch_max = torch.max(torch.abs(ttnn_result - torch_ref)).item()
        ttnn_vs_torch_avg = torch.mean(torch.abs(ttnn_result - torch_ref)).item()
        ttnn_vs_torch_ok = bool(torch.allclose(ttnn_result, torch_ref, rtol=1e-2, atol=1e-2))
        print(f"{'ttnn vs torch':<22} {str(ttnn_vs_torch_ok):<10} {ttnn_vs_torch_max:<14.6f} {ttnn_vs_torch_avg:<14.6f} {ttnn_ms:<10.2f}")

        t2 = time.time()
        y2_tt = MAKO_HOST_AND_KERNEL(a_tt)
        y2 = ttnn.to_torch(y2_tt, device=device)
        y2_ms = (time.time() - t2) * 1000.0
        y2_max = torch.max(torch.abs(y2 - ttnn_result)).item()
        y2_avg = torch.mean(torch.abs(y2 - ttnn_result)).item()
        y2_ok = bool(torch.allclose(y2, ttnn_result, rtol=1e-2, atol=1e-2))
        print(f"{'kernel vs ttnn':<22} {str(y2_ok):<10} {y2_max:<14.6f} {y2_avg:<14.6f} {y2_ms:<10.2f}")

    finally:
        ttnn.close_device(device)

if __name__ == "__main__":
    main()
