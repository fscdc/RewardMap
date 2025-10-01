from datasets import load_dataset
import os

# for training
dataset_name= "FSCCS/ReasonMap-Train"

# for evaluation
# dataset_name= "FSCCS/ReasonMap-Plus"

ds = load_dataset(dataset_name)

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", dataset_name.split('/')[-1])
os.makedirs(data_dir, exist_ok=True)

ds.save_to_disk(data_dir)
print(f"Dataset save to: {data_dir}")