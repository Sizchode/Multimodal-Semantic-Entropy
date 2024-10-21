import pandas as pd
import random

# Load the dataset
csv_file = "/users/zliu328/multimodal-semantics-entropy/filtered_iapr_dataset_77_tokens.csv"
df = pd.read_csv(csv_file)

# Shuffle the text descriptions (captions)
shuffled_captions = df['Description'].tolist()
random.shuffle(shuffled_captions)

# Ensure no image is paired with its correct caption
while any(df['Description'].tolist()[i] == shuffled_captions[i] for i in range(len(df))):
    random.shuffle(shuffled_captions)

# Create a new DataFrame with wrong pairs
df_wrong_pairs = df.copy()
df_wrong_pairs['Description'] = shuffled_captions

# Save the new wrong-pair dataset to a new CSV
output_csv = "/users/zliu328/multimodal-semantics-entropy/filtered_iapr_wrong_pairs.csv"
df_wrong_pairs.to_csv(output_csv, index=False)

print(f"Wrong-pair dataset saved to {output_csv}")
