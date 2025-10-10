# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
import ttnn._ttnn
from pathlib import Path
import sys
import random


def load_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

ROOT = Path.cwd()
EXAMPLES_DIR = ROOT / "examples" / "composable" / "hypot" 

# Load kernel sources
compute_hypot_src = load_file(EXAMPLES_DIR / "compute.cpp")
read_src = load_file(EXAMPLES_DIR / "read.cpp")
write_src = load_file(EXAMPLES_DIR / "write.cpp")


def hypot(input_tensor1: ttnn.Tensor, input_tensor2: ttnn.Tensor,) -> ttnn.Tensor:
    """
    Fused hypot implementation with feature flag to choose between:
    - include_sqrt=True: square(A) + square(B) + sqrt = hypot(A, B)
    - include_sqrt=False: square(A) + square(B) only
    """
    # Output tensor mirrors inputs
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(input_tensor1.shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        input_tensor1.device(),
    )

    # --- Tile count ---
    B, D = input_tensor1.shape
    
    # tiles count
    Mt = B // 32
    Nt = D // 32
    num_tiles = max(1, Mt * Nt)
    per_core_tile_cnt = num_tiles  # Keep naming consistent with square/sqrt wording
    
    # --- CB config (tile = 32x32 bf16) ---
    tile_bytes = 32 * 32 * 2  # bf16 = 2 bytes
    tiles_per_cb = 2
    cb_total = tiles_per_cb * tile_bytes
    cb_page_size = tile_bytes

    in1_cb, in2_cb, out_cb = 0, 1, 16
    scratch_cb_2, scratch_cb_3 = 2, 3  # Scratch CBs for square results
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # CB formats
    in1_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=in1_cb, data_format=ttnn.bfloat16, page_size=cb_page_size
    )
    in2_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=in2_cb, data_format=ttnn.bfloat16, page_size=cb_page_size
    )
    out_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=out_cb, data_format=ttnn.bfloat16, page_size=cb_page_size
    )
    scratch_cb_2_format = ttnn.CBFormatDescriptor(
        buffer_index=scratch_cb_2, data_format=ttnn.bfloat16, page_size=cb_page_size
    )
    scratch_cb_3_format = ttnn.CBFormatDescriptor(
        buffer_index=scratch_cb_3, data_format=ttnn.bfloat16, page_size=cb_page_size
    )

    # CB descriptors
    in1_cb_desc = ttnn.CBDescriptor(
        total_size=cb_total, core_ranges=core_grid, format_descriptors=[in1_cb_format]
    )
    in2_cb_desc = ttnn.CBDescriptor(
        total_size=cb_total, core_ranges=core_grid, format_descriptors=[in2_cb_format]
    )
    out_cb_desc = ttnn.CBDescriptor(
        total_size=cb_total, core_ranges=core_grid, format_descriptors=[out_cb_format]
    )
    scratch_cb_2_desc = ttnn.CBDescriptor(
        total_size=cb_total, core_ranges=core_grid, format_descriptors=[scratch_cb_2_format]
    )
    scratch_cb_3_desc = ttnn.CBDescriptor(
        total_size=cb_total, core_ranges=core_grid, format_descriptors=[scratch_cb_3_format]
    )

    # Compile-time args
    reader_ct_args = ttnn.TensorAccessorArgs(input_tensor1).get_compile_time_args()
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor2).get_compile_time_args())
    writer_ct_args = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()

    # Runtime args
    reader_rt_args = [[input_tensor1.buffer_address(), input_tensor2.buffer_address(), per_core_tile_cnt]]
    writer_rt_args = [[output_tensor.buffer_address(), num_tiles]]
    compute_ct = [num_tiles]  # per_core_tile_cnt as CT arg (index 0)
    compute_rt_args = []  # No runtime args for compute

    # Choose compute kernel based on feature flag
    compute_src = compute_hypot_src

    # Kernel descriptors
    reader_k = ttnn.KernelDescriptor(
        kernel_source=read_src,
        source_type=ttnn._ttnn.program_descriptor.SourceType.SOURCE_CODE,
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=[reader_rt_args],
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute_k = ttnn.KernelDescriptor(
        kernel_source=compute_src,  # Use the appropriate compute kernel
        source_type=ttnn._ttnn.program_descriptor.SourceType.SOURCE_CODE,
        core_ranges=core_grid,
        compile_time_args=compute_ct,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(),
    )
    writer_k = ttnn.KernelDescriptor(
        kernel_source=write_src,
        source_type=ttnn._ttnn.program_descriptor.SourceType.SOURCE_CODE,
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=[writer_rt_args],
        config=ttnn.WriterConfigDescriptor(),
    )

    # Program
    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_k, compute_k, writer_k],
        semaphores=[],
        cbs=[in1_cb_desc, in2_cb_desc, out_cb_desc, scratch_cb_2_desc, scratch_cb_3_desc],
    )

    # Execute
    return ttnn.generic_op([input_tensor1, input_tensor2, output_tensor], program_descriptor)

def get_inputs(case: int = 0):
    """
    Returns a square-ish tile-multiple shape (B, D) used for BOTH inputs A and B.
    """
    B = random.randint(1, 64)
    D = random.randint(1, 64)
    if case == 0:
        B = D = 1
    elif case == 1:
        B = 1; D = 2
    elif case == 2:
        B = 2; D = 1
    elif case == 3:
        B = 2; D = 2
    elif case == 4:
        B = D = 32
    elif case == 5:
        B = D = 64
    return (B, D)

def run():
    device = ttnn.open_device(device_id=0)
    case = -1  # Default case, can be changed
    size = get_inputs(case=case)
    torch.manual_seed(42)
    
    # two independent inputs with the SAME shape
    A = (torch.rand(size) - 0.5).to(torch.bfloat16)
    B = (torch.rand(size) - 0.5).to(torch.bfloat16)

    # move to TT
    A_tt = ttnn.from_torch(A, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    B_tt = ttnn.from_torch(B, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    print(f"Testing case {case} with shape {size}")
    print(f"Input A range: [{A.min():.4f}, {A.max():.4f}]")
    print(f"Input B range: [{B.min():.4f}, {B.max():.4f}]")
    print()

    # Test fused hypot
    print("Testing fused hypot...")
    Y_tt = hypot(A_tt, B_tt)
    Y = ttnn.to_torch(Y_tt, device=device)
    ref = torch.hypot(A.float(), B.float()).to(torch.bfloat16)
    
    max_err = torch.max(torch.abs(Y - ref))
    avg_err = torch.mean(torch.abs(Y - ref))
    allclose = torch.allclose(Y, ref, rtol=1e-2, atol=1e-2)
    
    print("max_err:", max_err)
    print("avg_err:", avg_err)
    print("allclose:", allclose)

def main():
    run()

if __name__ == "__main__":
    main()

