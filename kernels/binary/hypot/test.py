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

from examples.composable.hypot.host import hypot as main_hypot
from examples.composable.hypot.fused_square_add.host import fused_square_add_then_sqrt
from examples.composable.hypot.sequential.host import hypot_host as sequential_hypot

REF_FN = ttnn.hypot
SEQUENTIAL_FN = sequential_hypot
FUSED_SQUARE_ADD_THEN_SQRT = fused_square_add_then_sqrt
FULL_FUSED_FN = main_hypot

def main():
    torch.manual_seed(42)
    D = 128
    shape = (random.randint(1, D), random.randint(1, D))
    a = (torch.rand(shape, dtype=torch.bfloat16) - 0.5)  # ∈ [-0.5, 0.5)
    b = (torch.rand(shape, dtype=torch.bfloat16) - 0.5)  # ∈ [-0.5, 0.5)

    device = ttnn.open_device(device_id=0)

    try:

        """
        Key:
        - SEQUENTIAL is using host code using individual host codes for sqrt, matmul. hypot is computed using the host codes one by one
        - FUSED_SQ_ADD is uses fused kernels for square and add and then uses the host code for sqrt.
        - FULL_FUSED is uses fused kernels for square, add and sqrt.

        Accuracy:
        ------------------------------------------------------------
        out shape: (8, 71)
        Implementation         Allclose   Max Error      Avg Error      Time (ms) 
        ------------------------------------------------------------
        ttnn vs torch          True       0.003906       0.000290       2229.39   
        SEQUENTIAL vs ttnn     False      0.660156       0.166016       2074.35   
        FUSED_SQ_ADD vs ttnn   False      0.250000       0.116699       777.79    
        FULL_FUSED vs ttnn     False      0.498047       0.166016       857.19    

        -------------------------------------------------------------
        Note: For timings, FUSED_SQUARE_ADD_THEN_SQRT is much much faster if SEQUENTIAL_FN is run. Therefore these measurements we run 
        individually. These are not official timings, just a quick sanity check.
        

        Implementation         Allclose   Max Error      Avg Error      Time (ms) 
        ------------------------------------------------------------
        ttnn vs torch          True       0.003906       0.000332       2268.06  
        SEQUENTIAL vs ttnn     False      0.695312       0.118652       2106.08   
        FUSED_SQ_ADD vs ttnn   True       0.003906       0.001320       1491.42   
        FULL_FUSED vs ttnn     False      0.500000       0.155273       859.64    

        """

        print(f"shape: {shape}")
        print(f"A range: [{a.min().item():.4f}, {a.max().item():.4f}]")
        print(f"B range: [{b.min().item():.4f}, {b.max().item():.4f}]")
        print("Comparison to torch.hypot")
        print()

        a_tt = ttnn.from_torch(a, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        b_tt = ttnn.from_torch(b, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # Get torch reference
        torch_ref = torch.hypot(a.float(), b.float()).to(torch.bfloat16)
        
        t0 = time.time()
        ttnn_tt = REF_FN(a_tt, b_tt)
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

        t1 = time.time()
        y1_tt = SEQUENTIAL_FN(a_tt, b_tt)
        y1_ms = (time.time() - t1) * 1000.0

        y1 = ttnn.to_torch(y1_tt, device=device)
        y1_max = torch.max(torch.abs(y1 - ttnn_result)).item()
        y1_avg = torch.mean(torch.abs(y1 - ttnn_result)).item()
        y1_ok = bool(torch.allclose(y1, ttnn_result, rtol=1e-2, atol=1e-2))
        print(f"{'SEQUENTIAL vs ttnn':<22} {str(y1_ok):<10} {y1_max:<14.6f} {y1_avg:<14.6f} {y1_ms:<10.2f}")

        t2 = time.time()
        y2_tt = FUSED_SQUARE_ADD_THEN_SQRT(a_tt, b_tt)
        y2_ms = (time.time() - t2) * 1000.0
        y2 = ttnn.to_torch(y2_tt, device=device)
        y2_max = torch.max(torch.abs(y2 - ttnn_result)).item()
        y2_avg = torch.mean(torch.abs(y2 - ttnn_result)).item()
        y2_ok = bool(torch.allclose(y2, ttnn_result, rtol=1e-2, atol=1e-2))
        print(f"{'FUSED_SQ_ADD vs ttnn':<22} {str(y2_ok):<10} {y2_max:<14.6f} {y2_avg:<14.6f} {y2_ms:<10.2f}")

        t3 = time.time()
        y3_tt = FULL_FUSED_FN(a_tt, b_tt)
        y3_ms = (time.time() - t3) * 1000.0
        y3 = ttnn.to_torch(y3_tt, device=device)
        y3_max = torch.max(torch.abs(y3 - ttnn_result)).item()
        y3_avg = torch.mean(torch.abs(y3 - ttnn_result)).item()
        y3_ok = bool(torch.allclose(y3, ttnn_result, rtol=1e-2, atol=1e-2))
        print(f"{'FULL_FUSED vs ttnn':<22} {str(y3_ok):<10} {y3_max:<14.6f} {y3_avg:<14.6f} {y3_ms:<10.2f}")

    finally:
        ttnn.close_device(device)

if __name__ == "__main__":
    main()
