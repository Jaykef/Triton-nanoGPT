# Triton-Accelerated NanoGPT

## The WHY behind this ordeal

As part of my Triton challenge, after practicing the language for about 2 weeks, I attempted implementing custom Triton kernels for Karpathy's nanoGPT. This project serves as an eucational exploration of Triton's capabilities in accelerating transformer-based models.

### Key Features

1. Triton-accelerated kernels for key operations:
   - Softmax
   - Layer Normalization
   - GELU Activation
2. Modular architecture following the transformer design
3. Training and text generation capabilities
4. Visualization of training progress

## Implementation Details

### Model Architecture

The model follows the standard transformer architecture, including:

- Token and positional embeddings
- Multi-head attention mechanism
- Feedforward neural networks
- Layer normalization

### Triton Kernels

Custom Triton kernels have been implemented for:

1. Softmax computation
2. Layer normalization
3. GELU activation function

These kernels aim to optimize performance by leveraging GPU parallelism more effectively than standard PyTorch operations.

### Training

The training loop includes:

- Gradient accumulation for effective larger batch sizes
- Learning rate scheduling
- Gradient clipping
- Validation loss tracking

## How to Use

1. **Setup**: Requires GPU! Ensure you have PyTorch and Triton installed.

2. **Data Preparation**: The code uses the Tiny Shakespeare dataset by default. It will be downloaded automatically if not present.

3. **Training**: 
   ```python
   python triton_train_nanoGPT.py
   ```

## License

MIT
