import argparse
import random
import torch
from torch.utils.data import Dataset, DataLoader
import open_clip
from PIL import Image
from tqdm import tqdm
import pandas as pd

# Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the dataset class
class ContrastiveClipDataset(Dataset):
    def __init__(self, csv_file, preprocess, tokenizer):
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.data = pd.read_csv(csv_file)

        # Ensure correct columns
        if 'Image File Path' in self.data.columns and 'Description' in self.data.columns:
            self.images = self.data['Image File Path'].tolist()
            self.captions = self.data['Description'].tolist()
        else:
            raise KeyError("Expected columns 'Image File Path' and 'Description' not found in CSV file.")
        
        # Create shuffled captions for negative pairs, ensuring no correct pairs are retained
        self.shuffled_captions = random.sample(self.captions, len(self.captions))
        while any(a == b for a, b in zip(self.captions, self.shuffled_captions)):
            self.shuffled_captions = random.sample(self.captions, len(self.captions))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        caption = self.captions[idx]  # Correct caption
        wrong_caption = self.shuffled_captions[idx]  # Incorrect caption

        # Open and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)

        # Tokenize both the correct and incorrect captions
        correct_text = self.tokenizer(caption)
        wrong_text = self.tokenizer(wrong_caption)

        return image, correct_text, wrong_text

# Fine-tuning the CLIP model with gradient accumulation
def fine_tune_clip(csv_file, model_save_path, num_epochs, learning_rate, batch_size, accumulation_steps):
    # Load the OpenCLIP model and tokenizer
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.to(device)

    # Prepare dataset and dataloader
    dataset = ContrastiveClipDataset(csv_file, preprocess, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        optimizer.zero_grad()  # Start with cleared gradients

        for images, correct_texts, wrong_texts in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            # Accumulation step over mini-batches
            for i in range(accumulation_steps):
                # Forward pass
                batch_images = images.to(device)
                batch_correct_texts = correct_texts.squeeze(1).to(device)
                batch_wrong_texts = wrong_texts.squeeze(1).to(device)

                # Encode images and both correct/incorrect texts
                image_features = model.encode_image(batch_images)
                correct_text_features = model.encode_text(batch_correct_texts)
                wrong_text_features = model.encode_text(batch_wrong_texts)

                # Normalize the features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                correct_text_features = correct_text_features / correct_text_features.norm(dim=-1, keepdim=True)
                wrong_text_features = wrong_text_features / wrong_text_features.norm(dim=-1, keepdim=True)

                # Compute similarity logits for correct and wrong pairs
                correct_logits = (image_features @ correct_text_features.t()) * model.logit_scale.exp()
                wrong_logits = (image_features @ wrong_text_features.t()) * model.logit_scale.exp()

                # Contrastive loss: Maximize correct pair similarities, minimize wrong pair similarities
                loss = -(torch.log_softmax(correct_logits, dim=1).diag().mean() -
                         torch.log_softmax(wrong_logits, dim=1).diag().mean())

                # Accumulate gradients
                loss.backward()

                if (i + 1) % accumulation_steps == 0:  # Apply optimizer after accumulation steps
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}")

    # Dynamically construct the save path with hyperparameters
    model_save_path = f"{model_save_path}_lr{learning_rate}_bs{batch_size}_epochs{num_epochs}.pth"
    
    # Save the fine-tuned model
    torch.save(model.state_dict(), model_save_path)
    print(f"Fine-tuned model saved to {model_save_path}")

# Tentative ! wrong_pair fine tune
def wrong_pair_fine_tune_clip(csv_file, model_save_path, num_epochs, learning_rate, batch_size, accumulation_steps):
    # Load the OpenCLIP model and tokenizer
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.to(device)

    # Prepare dataset and dataloader
    dataset = ContrastiveClipDataset(csv_file, preprocess, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        optimizer.zero_grad()  # Start with cleared gradients

        for images, correct_texts, wrong_texts in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            # Accumulation step over mini-batches
            for i in range(accumulation_steps):
                # Forward pass
                batch_images = images.to(device)
                # Swapping correct and wrong text to fine-tune with wrong image-text pairs as correct
                batch_correct_texts = wrong_texts.squeeze(1).to(device)
                batch_wrong_texts = correct_texts.squeeze(1).to(device)

                # Encode images and both wrong/correct texts (swapped)
                image_features = model.encode_image(batch_images)
                correct_text_features = model.encode_text(batch_correct_texts)
                wrong_text_features = model.encode_text(batch_wrong_texts)

                # Normalize the features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                correct_text_features = correct_text_features / correct_text_features.norm(dim=-1, keepdim=True)
                wrong_text_features = wrong_text_features / wrong_text_features.norm(dim=-1, keepdim=True)

                # Compute similarity logits for swapped pairs
                correct_logits = (image_features @ correct_text_features.t()) * model.logit_scale.exp()
                wrong_logits = (image_features @ wrong_text_features.t()) * model.logit_scale.exp()

                # Contrastive loss: Maximize swapped pair similarities, minimize correct pairs similarities (as wrong)
                loss = -(torch.log_softmax(correct_logits, dim=1).diag().mean() -
                         torch.log_softmax(wrong_logits, dim=1).diag().mean())

                # Accumulate gradients
                loss.backward()

                if (i + 1) % accumulation_steps == 0:  # Apply optimizer after accumulation steps
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}")

    # Dynamically construct the save path with hyperparameters
    model_save_path = f"{model_save_path}_wrongpair_lr{learning_rate}_bs{batch_size}_epochs{num_epochs}.pth"
    
    # Save the fine-tuned model
    torch.save(model.state_dict(), model_save_path)
    print(f"Fine-tuned model saved to {model_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune CLIP model")
    
    # Add arguments for hyperparameters only
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of epochs for training")
    parser.add_argument('--learning_rate', type=float, default=5e-05, help="Learning rate for the optimizer")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training")
    parser.add_argument('--accumulation_steps', type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument('--wrong_pair', action='store_true', help="Flag to fine-tune with wrong pairs as correct")

    args = parser.parse_args()
    
    # Hardcoded file paths
    csv_file = "/users/zliu328/multimodal-semantics-entropy/filtered_iapr_dataset_77_tokens.csv"
    model_save_path = "/users/zliu328/multimodal-semantics-entropy/clip_finetuned_rightpair"
    
    # Check if wrong_pair flag is passed, then call the respective fine-tuning function
    if args.wrong_pair:
It seems like the message was cut off. Here's how you can finish the conditional check and choose between the normal fine-tuning and the "wrong pair" fine-tuning based on the `--wrong_pair` flag:

```python
    # Check if wrong_pair flag is passed, then call the respective fine-tuning function
    if args.wrong_pair:
        # Call the wrong pair fine-tuning function
        wrong_pair_fine_tune_clip(csv_file, model_save_path, args.num_epochs, args.learning_rate, args.batch_size, args.accumulation_steps)
    else:
        # Call the regular fine-tuning function
        fine_tune_clip(csv_file, model_save_path, args.num_epochs, args.learning_rate, args.batch_size, args.accumulation_steps)
