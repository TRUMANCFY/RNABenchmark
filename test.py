import os, sys
import torch
from model.rnalm.rnalm_config import RnaLmConfig

current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)

from model.rnalm.modeling_rnalm import RnaLmModel
from tokenizer.tokenization_opensource import OpenRnaLMTokenizer

ckpt = "./checkpoint/baseline/BEACON-B"

# config = RnaLmConfig.from_pretrained(ckpt)

# print(config)

# 2) Load tokenizer + model with the patched config
tokenizer = OpenRnaLMTokenizer.from_pretrained(
    ckpt,
    model_max_length=512,
    padding_side="right",
    use_fast=True,
)

model = RnaLmModel.from_pretrained(ckpt, trust_remote_code=True,)
model.eval()

# 3) Run a forward pass
sequences = ["AUUCCGAUUCCGAUUCCG"]
batch = tokenizer(
    sequences,
    return_tensors="pt",
    padding="longest",
    max_length=1026,
    truncation=True,
)

with torch.no_grad():
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    # outputs may be a tuple; take last_hidden_state if available
    embedding = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.last_hidden_state

print(embedding.shape)
