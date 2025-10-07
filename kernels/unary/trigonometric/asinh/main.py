# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
import ttnn._ttnn

def load_file(p): 
    with open(p, "r") as f: 
        return f.read()

READ_SRC_PATH    = "/home/george/tt-generations/data/manual/eltwise_copy/read.cpp"
WRITE_SRC_PATH   = "/home/george/tt-generations/data/manual/eltwise_copy/write.cpp"
COMPUTE_SRC_PATH = "/home/george/tt-generations/data/manual/eltwise_copy/compute.cpp"  

read_src  = load_file(READ_SRC_PATH)
write_src = load_file(WRITE_SRC_PATH)
comp_src  = load_file(COMPUTE_SRC_PATH)

def compute(x: ttnn.Tensor) -> ttnn.Tensor:
    # Output mirrors input
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape(x.shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, x.device()
    )

    # CBs
    tile_bytes   = 32*32*2
    tiles_per_cb = 2
    total_bytes  = tiles_per_cb * tile_bytes
    cb_in, cb_out = 0, 16

    in_fmt  = ttnn.CBFormatDescriptor(buffer_index=cb_in,  data_format=ttnn.bfloat16, page_size=tile_bytes)
    out_fmt = ttnn.CBFormatDescriptor(buffer_index=cb_out, data_format=ttnn.bfloat16, page_size=tile_bytes)

    core = ttnn.CoreCoord(0, 0)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    in_cb  = ttnn.CBDescriptor(total_size=total_bytes, core_ranges=grid, format_descriptors=[in_fmt])
    out_cb = ttnn.CBDescriptor(total_size=total_bytes, core_ranges=grid, format_descriptors=[out_fmt])

    # tiles count
    B, D = x.shape
    assert B % 32 == 0 and D % 32 == 0, "Pad to multiples of 32"
    num_tiles = (B // 32) * (D // 32)

    # CT/RT args
    reader_ct = ttnn.TensorAccessorArgs(x).get_compile_time_args()
    writer_ct = ttnn.TensorAccessorArgs(y).get_compile_time_args()
    reader_rt = [[x.buffer_address(),  num_tiles]]
    writer_rt = [[y.buffer_address(),  num_tiles]]
    compute_ct = [num_tiles]  # per_core_tile_cnt
    compute_rt = []

    reader_k = ttnn.KernelDescriptor(
        kernel_source=read_src,
        source_type=ttnn._ttnn.program_descriptor.SourceType.SOURCE_CODE,
        core_ranges=grid,
        compile_time_args=reader_ct,
        runtime_args=[reader_rt],
        config=ttnn.ReaderConfigDescriptor(),
    )
    compute_k = ttnn.KernelDescriptor(
        kernel_source=comp_src,
        source_type=ttnn._ttnn.program_descriptor.SourceType.SOURCE_CODE,
        core_ranges=grid,
        compile_time_args=compute_ct,
        runtime_args=[compute_rt],
        config=ttnn.ComputeConfigDescriptor(),
    )
    writer_k = ttnn.KernelDescriptor(
        kernel_source=write_src,
        source_type=ttnn._ttnn.program_descriptor.SourceType.SOURCE_CODE,
        core_ranges=grid,
        compile_time_args=writer_ct,
        runtime_args=[writer_rt],
        config=ttnn.WriterConfigDescriptor(),
    )

    prog = ttnn.ProgramDescriptor(
        kernels=[reader_k, compute_k, writer_k],
        semaphores=[],
        cbs=[in_cb, out_cb],
    )

    return ttnn.generic_op([x, y], prog)

def main():
    dev = ttnn.open_device(device_id=0)
    X = (torch.rand(64, 64) - 0.5).to(torch.bfloat16)  # 2x2 tiles, any real is fine
    Xtt = ttnn.from_torch(X, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    Ytt = compute(Xtt)
    Y = ttnn.to_torch(Ytt, device=dev)
    ref = torch.asinh(X).to(torch.bfloat16)
    breakpoint()
    print("max_err:", torch.max(torch.abs(Y - ref)))
    print("allclose:", torch.allclose(Y, ref, rtol=1e-2, atol=1e-2))

if __name__ == "__main__":
    main()
