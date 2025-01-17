import torch
import time
from torch.utils.cpp_extension import load

# Disable gradient calculations for efficiency
torch.set_grad_enabled(False)

# Load the CUDA extension (ensure 'histogram.cu' contains the corrected C++/CUDA code)
histogram = load(
    name="histogram",
    sources=["histogram.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=['-std=c++17']
)

def cpu_histogram(a, histogram_size=256):
    """
    Compute the histogram on the CPU using torch.bincount.

    Args:
        a (torch.Tensor): 1D tensor of integers.
        histogram_size (int): The size of the histogram (number of bins).

    Returns:
        torch.Tensor: Histogram tensor of shape (histogram_size,).
    """
    # Ensure that all values in 'a' are within [0, histogram_size - 1]
    assert torch.all((a >= 0) & (a < histogram_size)), "Values in 'a' are out of the valid range."

    # Compute histogram using torch.bincount for efficiency
    hist = torch.bincount(a, minlength=histogram_size)

    # If 'hist' is shorter than 'histogram_size', pad it with zeros
    if hist.numel() < histogram_size:
        padding = torch.zeros(histogram_size - hist.numel(), dtype=hist.dtype, device=hist.device)
        hist = torch.cat([hist, padding], dim=0)

    return hist

def cpu_histogram_pure_python(a, histogram_size=256):
    """
    Compute the histogram on the CPU using pure Python.

    Args:
        a (torch.Tensor): 1D tensor of integers.
        histogram_size (int): The size of the histogram (number of bins).

    Returns:
        torch.Tensor: Histogram tensor of shape (histogram_size,).
    """
    # Initialize histogram with zeros
    hist = [0] * histogram_size

    # Iterate through each element and increment the corresponding bin
    for value in a.tolist():
        if 0 <= value < histogram_size:
            hist[value] += 1
        else:
            raise ValueError(f"Value {value} in 'a' is out of the valid range [0, {histogram_size - 1}].")

    return torch.tensor(hist, dtype=torch.int32, device=a.device)

if __name__ == "__main__":
    # Define histogram range
    start_idx = 0
    end_idx = 256  # Histogram bins: 0 to 255

    # Set random seed for reproducibility
    torch.manual_seed(1)

    # Initialize input tensor 'a' with random integers in [start_idx, end_idx)
    a = torch.randint(start_idx, end_idx, (end_idx,), dtype=torch.int32).cuda()

    # Initialize output tensors 'b' and 'b2' with zeros (dtype matches CUDA kernels)
    b = torch.zeros([end_idx], dtype=torch.int32).cuda()
    b2 = torch.zeros([end_idx], dtype=torch.int32).cuda()

    # Compute ground truth histogram on the CPU using torch.bincount
    ground_truth = cpu_histogram(a.cpu(), histogram_size=end_idx).cuda()

    # Alternatively, use the pure Python implementation
    # ground_truth = cpu_histogram_pure_python(a.cpu(), histogram_size=end_idx).cuda()

    # Launch CUDA histogram kernels
    histogram.histogram_i32_launcher(a, b)
    histogram.histogram_i32x4_launcher(a, b2)

    # Wait for CUDA kernels to finish
    torch.cuda.synchronize()

    # Move histograms to CPU for comparison
    b_cpu = b.cpu()
    b2_cpu = b2.cpu()
    ground_truth_cpu = ground_truth.cpu()

    # Print a subset of the histograms for verification
    print("CPU Ground Truth Histogram (first 10 bins):")
    print(ground_truth_cpu.tolist()[:10])

    print("\nCUDA Kernel 1 Histogram (b) - First 10 bins:")
    print(b_cpu.tolist()[:10])

    print("\nCUDA Kernel 2 Histogram (b2) - First 10 bins:")
    print(b2_cpu.tolist()[:10])

    # Verify that both CUDA kernels match the ground truth
    if torch.equal(b_cpu, ground_truth_cpu):
        print("\n✅ CUDA Kernel 1 matches the CPU ground truth.")
    else:
        print("\n❌ CUDA Kernel 1 does NOT match the CPU ground truth.")

    if torch.equal(b2_cpu, ground_truth_cpu):
        print("✅ CUDA Kernel 2 matches the CPU ground truth.")
    else:
        print("❌ CUDA Kernel 2 does NOT match the CPU ground truth.")

    # Optional: Assert to enforce correctness
    assert torch.equal(b_cpu, ground_truth_cpu), "CUDA Kernel 1 histogram does not match the ground truth."
    assert torch.equal(b2_cpu, ground_truth_cpu), "CUDA Kernel 2 histogram does not match the ground truth."

    print("\n🎉 All histograms match the ground truth!")

