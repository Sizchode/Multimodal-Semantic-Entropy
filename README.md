# Multimodal-Semantic-Entropy

## Hyper-parameters

- **Model**: OpenCLIP ViT-B/32
- **Dataset**: Filtered IAPR Dataset (17,000 image-text pairs)
- **Batch Size**: Entire dataset (17,000), split into 2 mini-batches with gradient accumulation
- **Effective Batch Size**: 34,000 (Mini-batch size of 17,000, accumulated over 2 steps)
- **Learning Rate**: 5e-05
- **Gradient Accumulation Steps**: 2
- **Number of Epochs**: 20
- **Optimizer**: AdamW
- **Loss Function**: Contrastive Loss (correct and wrong pairs similarity)
- **Device**: GPU (with CUDA support)
