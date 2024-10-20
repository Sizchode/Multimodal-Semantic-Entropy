# Multimodal-Semantic-Entropy

## Hyper-parameters

- **Model**: OpenCLIP ViT-B/32
- **Dataset**: Filtered IAPR Dataset (17k image-text pairs)
- **Batch Size**: 128, split into 2 mini-batches with gradient accumulation
- **Effective Batch Size**: 256 (Mini-batch size of 17,000, accumulated over 2 steps)
- **Learning Rate**: 5e-05
- **Gradient Accumulation Steps**: 2
- **Number of Epochs**: 20
- **Optimizer**: AdamW
- **Loss Function**: Contrastive Loss (correct and wrong pairs similarity)
- **Device**: GPU (with CUDA support)
- **Command** python fine-tune.py --batch_size 128 --num_epochs 20 --learning_rate 5e-5 --accumulation_steps 2
- **Command for wrong pair** python fine-tune.py --num_epochs 20 --learning_rate 5e-05 --batch_size 128 --accumulation_steps 2 --wrong_pair

