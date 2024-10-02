# Triton-Accelerated NanoGPT

#### The WHY behind this ordeal

As part of my Triton challenge, after practicing the language for about 2 weeks, I attempted implementing custom Triton kernels for Karpathy's nanoGPT. This project serves as an eucational exploration of Triton's capabilities in accelerating transformer-based models.

### Key Features

1. Triton-accelerated kernels for key ml ops:
   - Softmax
   - Layer Normalization
   - GELU Activation
2. Modular architecture following a standard transformer
3. Text generation

### Training

GPU-aware train loop with effective gradient accumulation, learning rate scheduling and gradient clipping with val loss tracking.


### How to Use

1. **Setup**: Requires GPU! Ensure you have PyTorch and Triton installed. GPU Poor? I am too, I used a google colab.

2. **Data Preparation**: Using Tiny Shakespeare dataset by default. It will be downloaded automatically if not present.

3. **Training**: 
   ```python
   python triton_train_nanoGPT.py
   ```
This will train for 50 epochs, save checkpoint as `nanoGPT_cpkt.pth` and sample from it.
## License

MIT
