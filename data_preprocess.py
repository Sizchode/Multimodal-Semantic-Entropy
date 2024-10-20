import pandas as pd
import xml.etree.ElementTree as ET

# Paths to the datasets
member_data_file = '/users/zliu328/iapr/iapr_training_dataset.csv'
non_member_data_file = '/users/zliu328/iapr/iapr_nonmembership_dataset.csv'

# Output CSV file path
output_csv_file = '/users/zliu328/multimodal-semantics-entropy/filtered_iapr_dataset.csv'

# Function to extract the <DESCRIPTION> from an annotation file
def extract_description(annotation_file_path):
    try:
        tree = ET.parse(annotation_file_path)
        root = tree.getroot()

        # Extract the <DESCRIPTION> tag's content
        description = root.findtext('DESCRIPTION')

        # Return description if valid, otherwise None
        if description and description.strip():
            return description.strip()
        else:
            return None
    except (ET.ParseError, FileNotFoundError) as e:
        print(f"Error processing file {annotation_file_path}: {e}")
        return None

# Read the member and non-member datasets
member_df = pd.read_csv(member_data_file)
non_member_df = pd.read_csv(non_member_data_file)

# Combine the two datasets
combined_df = pd.concat([member_df, non_member_df], ignore_index=True)

# List to store valid image paths and descriptions
valid_data = []

# Iterate over the dataset and filter based on the <DESCRIPTION>
for idx, row in combined_df.iterrows():
    image_path = row['Image File Path']
    annotation_path = row['Annotation File Path']
    
    # Extract description from the annotation file
    description = extract_description(annotation_path)
    
    # If a valid description exists, add it to the valid data list
    if description:
        valid_data.append({'Image File Path': image_path, 'Description': description})

# Create a DataFrame with the valid data
filtered_df = pd.DataFrame(valid_data)

# Save the filtered dataset to a new CSV file
filtered_df.to_csv(output_csv_file, index=False)

# filtered_df.head()  # Display the first few rows of the filtered data

import pandas as pd
import open_clip

# Path to the existing filtered dataset
csv_file = '/users/zliu328/multimodal-semantics-entropy/filtered_iapr_dataset.csv'

# Output CSV file path for the 77-token filtered dataset
output_csv_file = '/users/zliu328/multimodal-semantics-entropy/filtered_iapr_dataset_77_tokens.csv'

# Define the max token limit for CLIP
MAX_TOKENS = 77

# Load the OpenCLIP tokenizer
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Load the existing dataset
df = pd.read_csv(csv_file)

# List to store valid image paths and descriptions
valid_data = []

# Iterate over the dataset and filter based on token length
for idx, row in df.iterrows():
    image_path = row['Image File Path']
    description = row['Description']
    
    # Tokenize the description using CLIP tokenizer
    tokenized_description = tokenizer(description)
    
    # Check if token length exceeds the maximum allowed length
    if tokenized_description.shape[1] <= MAX_TOKENS:  # CLIP tokenizer returns (batch_size, seq_len)
        valid_data.append({'Image File Path': image_path, 'Description': description})

# Create a DataFrame with the valid data
filtered_df = pd.DataFrame(valid_data)

# Save the filtered dataset to a new CSV file
filtered_df.to_csv(output_csv_file, index=False)

# Print the first few rows of the filtered dataset and count the valid pairs
print(filtered_df.head())
print(f"Number of valid image-text pairs after 77-token filtering: {len(filtered_df)}")
