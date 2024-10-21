# Triton-Accelerated NanoGPT

The WHY behind this ordeal

After practicing triton for about 2 weeks now, I challenged myself into implementing custom triton kernels for Karpathy's nanoGPT and quite an ordeal it was but somehow got something working, not perfect but getting there:), contributions are welcomed.

## Kernels
Supports lightweight custom triton kernels for softmax, layer normalization, cross entropy loss and GELU activation.

## Training

GPU-aware train loop with effective gradient accumulation, learning rate scheduling and gradient clipping with val loss tracking.

- **Setup**: Requires GPU! Ensure you have PyTorch and Triton installed. GPU Poor? I am too, I used one free T4 on google colab.

- **Data**: Using Tiny Shakespeare dataset by default. It will be downloaded automatically if not present.

- **Training**: 
   ```python
   python triton_nanoGPT.py
   ```
This will train for 100 epochs, save checkpoint as `nanoGPT_cpkt.pth` and sample from it.
## License

MIT
