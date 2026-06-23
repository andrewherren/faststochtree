# CUDA reprexes — `reprex.ipynb`

The CUDA counterparts of the Metal reprexes in [`../metal/`](../metal/), packaged
as a single Colab notebook so no local NVIDIA GPU is needed.

## Run it

1. Open `reprex.ipynb` in [Google Colab](https://colab.research.google.com/)
   (`File → Upload notebook`, or click the "Open in Colab" button below).
2. **Runtime → Change runtime type → GPU** (a free T4 is plenty).
3. Run all cells top to bottom.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrewherren/faststochtree/blob/main/experimental/gpu/cuda/reprex.ipynb)

Each benchmark is written to a `.cu` file by a `%%writefile` cell and compiled with
`nvcc` in the next cell — nothing to install beyond what Colab's GPU runtime ships.

## What's inside

| Cell | Measures | Metal twin |
| --- | --- | --- |
| **1. launch_overhead** | empty-kernel launch cost: naive `<<<>>>` vs CUDA Graph replay | `dispatch_overhead` |
| **2. grid_barrier** | cost of the hardware `grid.sync()` device-wide barrier, swept over block count (stays cheap and flat) | `grid_barrier` |

Expected results and the cross-platform interpretation are in the notebook's
markdown cells and in [`../README.md`](../README.md).
