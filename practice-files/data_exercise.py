# exercise 1
from datasets import load_dataset
import time
import os

data_1 = load_dataset("glue", "mrpc", split="train")
first_5 = data_1[:5]

for i, v in first_5.items():
    print(i, v)

# exercise 2: streaming
data_c4 = load_dataset("allenai/c4", "en", streaming=True, split="train")
start = time.time()
count = 0

for example in data_c4:
    if time.time() - start >= 10:
        break
    count += 1
print(f" Processed {count} examples in 10 seconds")

# exercise 3: checking the size of dataset in different format

data_1.to_csv("/home/strawhat/ai-engineering-from-scratch/data/glue.csv")
data_1.to_parquet("/home/strawhat/ai-engineering-from-scratch/data/glue.parquet")
csv_size = os.path.getsize("/home/strawhat/ai-engineering-from-scratch/data/glue.csv")
parquet_size = os.path.getsize("/home/strawhat/ai-engineering-from-scratch/data/glue.parquet")

print(f"csv size is {csv_size / 1024:.2f}KB")
print(f"Parquet size is : {parquet_size / 1024:.2f}KB")


# exercise 4: splitting
split = data_1.train_test_split(test_size=0.3, seed=42)
temp = split["test"]
train = split["train"]

val_test = temp.train_test_split(test_size=0.5, seed=42)
val = val_test["train"]
test = val["test"]
print(f" total size : {data_1.shape}, train size: {train.shape}, val size: {val.shape}, Test size: {test.shape}")

