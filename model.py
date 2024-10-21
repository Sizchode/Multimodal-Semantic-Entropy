import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import open_clip
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F

# Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the dataset class
class ContrastiveClipDataset(Dataset):
    def __init__(self, csv_file, preprocess, tokenizer):
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['Image File Path']
        caption = self.data.iloc[idx]['Description']

        # Open and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)

        # Tokenize the caption
        text = self.tokenizer(caption)

        return image, text


# Fine-tuning function for right-pair
def fine_tune_clip(right_csv, wrong_csv, model_save_path, num_epochs, learning_rate, batch_size, accumulation_steps):
    # Load the OpenCLIP model and tokenizer
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.to(device)

    # Prepare right-pair and wrong-pair datasets and dataloaders
    right_pair_dataset = ContrastiveClipDataset(right_csv, preprocess, tokenizer)
    wrong_pair_dataset = ContrastiveClipDataset(wrong_csv, preprocess, tokenizer)

    right_dataloader = DataLoader(right_pair_dataset, batch_size=batch_size, shuffle=False)
    wrong_dataloader = DataLoader(wrong_pair_dataset, batch_size=batch_size, shuffle=False)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        optimizer.zero_grad()  # Start with cleared gradients

        for step, (right_batch, wrong_batch) in enumerate(tqdm(zip(right_dataloader, wrong_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", total=len(right_dataloader))):
            right_images, right_texts = right_batch
            _, wrong_texts = wrong_batch

            # Forward pass
            right_images = right_images.to(device)
            right_texts = right_texts.squeeze(1).to(device)
            wrong_texts = wrong_texts.squeeze(1).to(device)

            # Encode images and texts
            image_features = model.encode_image(right_images)
            right_text_features = model.encode_text(right_texts)
            wrong_text_features = model.encode_text(wrong_texts)

            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            right_text_features = right_text_features / right_text_features.norm(dim=-1, keepdim=True)
            wrong_text_features = wrong_text_features / wrong_text_features.norm(dim=-1, keepdim=True)

            # Compute similarity logits for right and wrong pairs
            right_logits = image_features @ right_text_features.t() * model.logit_scale.exp()
            wrong_logits = image_features @ wrong_text_features.t() * model.logit_scale.exp()

            # Define target labels for correct pairs
            target = torch.arange(right_logits.shape[0], device=device)

            # Compute Cross-Entropy loss
            loss = (F.cross_entropy(right_logits, target) + F.cross_entropy(wrong_logits, target)) / 2

            # Accumulate gradients
            loss.backward()

            # Apply optimizer after the accumulation steps
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}")

    # Save the fine-tuned model
    model_save_path = f"{model_save_path}_lr{learning_rate}_bs{batch_size}_epochs{num_epochs}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Fine-tuned model saved to {model_save_path}")


# Fine-tuning function for wrong-pair
def fine_tune_wrong_pair(right_csv, wrong_csv, model_save_path, num_epochs, learning_rate, batch_size, accumulation_steps):
    # Same as the fine_tune_clip function but roles are reversed

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.to(device)

    right_pair_dataset = ContrastiveClipDataset(right_csv, preprocess, tokenizer)
    wrong_pair_dataset = ContrastiveClipDataset(wrong_csv, preprocess, tokenizer)

    right_dataloader = DataLoader(right_pair_dataset, batch_size=batch_size, shuffle=False)
    wrong_dataloader = DataLoader(wrong_pair_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        optimizer.zero_grad()

        for step, (right_batch, wrong_batch) in enumerate(tqdm(zip(right_dataloader, wrong_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", total=len(right_dataloader))):
            right_images, wrong_texts = right_batch
            _, right_texts = wrong_batch

            right_images = right_images.to(device)
            wrong_texts = wrong_texts.squeeze(1).to(device)
            right_texts = right_texts.squeeze(1).to(device)

            image_features = model.encode_image(right_images)
            wrong_text_features = model.encode_text(wrong_texts)
            right_text_features = model.encode_text(right_texts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            wrong_text_features = wrong_text_features / wrong_text_features.norm(dim=-1, keepdim=True)
            right_text_features = right_text_features / right_text_features.norm(dim=-1, keepdim=True)

            wrong_logits = image_features @ wrong_text_features.t() * model.logit_scale.exp()
            right_logits = image_features @ right_text_features.t() * model.logit_scale.exp()

            target = torch.arange(wrong_logits.shape[0], device=device)

            loss = (F.cross_entropy(wrong_logits, target) + F.cross_entropy(right_logits, target)) / 2

            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}")

    model_save_path = f"{model_save_path}_wrong_pair_lr{learning_rate}_bs{batch_size}_epochs{num_epochs}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Wrong-pair fine-tuned model saved to {model_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune CLIP model")

    parser.add_argument('--num_epochs', type=int, default=20, help="Number of epochs for training")
    parser.add_argument('--learning_rate', type=float, default=5e-05, help="Learning rate for the optimizer")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training")
    parser.add_argument('--accumulation_steps', type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument('--fine_tune_type', type=str, choices=['right_pair', 'wrong_pair', 'both'], default='right_pair', help="Type of fine-tuning: right_pair, wrong_pair, or both")

    args = parser.parse_args()

    csv_file = "/users/zliu328/multimodal-semantics-entropy/filtered_iapr_dataset_77_tokens.csv"
    wrong_csv = "/users/zliu328/multimodal-semantics-entropy/filtered_iapr_wrong_pairs.csv"
    model_save_path = "/users/zliu328/multimodal-semantics-entropy/clip_finetuned"

    if args.fine_tune_type == 'right_pair':
        fine_tune_clip(csv_file, wrong_csv, model_save_path, args.num_epochs, args.learning_rate, args.batch_size, args.accumulation_steps)
    elif args.fine_tune_type == 'wrong_pair':
        fine_tune_wrong_pair(csv_file, wrong_csv, model_save_path, args.num_epochs, args.learning_rate, args.batch_size, args.accumulation_steps)
    elif args.fine_tune_type == 'both':
        fine_tune_clip(csv_file, wrong_csv, model_save_path, args.num_epochs, args.learning_rate, args.batch_size, args.accumulation_steps)
        fine_tune_wrong_pair(csv_file, wrong_csv, model_save_path, args.num_epochs, args.learning_rate, args.batch_size, args.accumulation_steps) 
