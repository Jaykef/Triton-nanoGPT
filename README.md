# Triton-Accelerated NanoGPT

The WHY behind this ordeal

After practicing triton for about 2 weeks, I attempted implementing custom Triton kernels for Karpathy's nanoGPT. This project serves as an eucational exploration of Triton's capabilities in accelerating transformer-based models. It's not perfect and would apprecieate contributions.

## Kernels
Supports custom Triton-accelerated kernels for softmax, layer normalization and GELU activation.

## Training

GPU-aware train loop with effective gradient accumulation, learning rate scheduling and gradient clipping with val loss tracking.

- **Setup**: Requires GPU! Ensure you have PyTorch and Triton installed. GPU Poor? I am too, I used one free T4 on google colab.

- **Data**: Using Tiny Shakespeare dataset by default. It will be downloaded automatically if not present.

- **Training**: 
   ```python
   python triton_train_nanoGPT.py
   ```
This will train for 50 epochs, save checkpoint as `nanoGPT_cpkt.pth` and sample from it.
## License

MIT
