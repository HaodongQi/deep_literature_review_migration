
# how to use a GPU for testing or building llama.cpp
# salloc -p gpu_test --gres=gpu:1 --mem=40G -N 1 -t 60
# module load nvhpc/23.7-fasrc01
# module load cuda/12.4.1-fasrc01
# conda activate jago
# then execute this script

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import torch

# 50% train
base_model = "meta-llama/Llama-3.2-3B-Instruct" # this must leave in HF
new_model = "local/Llama-32-3B-climate-fasrc"  # this can leave in huggingface
local_model = "Llama-32-3B-climate-fasrc" # this is the merged model saved locally in the dir "models"


model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map={"": 0},
)

finetuned_model = PeftModel.from_pretrained(model, new_model)
finetuned_model.save_pretrained("adapters/Llama-32-3B-climate-fasrc")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = "right"
tokenizer.save_pretrained("adapters/Llama-32-3B-climate-fasrc")
merged_model = finetuned_model.merge_and_unload()
merged_model.save_pretrained(local_model)
tokenizer.save_pretrained(local_model)

# 80 % train

base_model = "meta-llama/Llama-3.2-3B-Instruct" # this must leave in HF
new_model = "local/Llama-32-3B-80-climate-fasrc"  # this can leave in huggingface
local_model = "Llama-32-3B-80-climate-fasrc" # this is the merged model saved locally in the dir "models"


model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map={"": 0},
)

finetuned_model = PeftModel.from_pretrained(model, new_model)
finetuned_model.save_pretrained("adapters/Llama-32-3B-80-climate-fasrc")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = "right"
tokenizer.save_pretrained("adapters/Llama-32-3B-80-climate-fasrc")
merged_model = finetuned_model.merge_and_unload()
merged_model.save_pretrained(local_model)
tokenizer.save_pretrained(local_model)



# 95 % train


base_model = "meta-llama/Llama-3.2-3B-Instruct" # this must leave in HF
new_model = "local/Llama-32-3B-95-climate-fasrc"  # this can leave in huggingface
local_model = "Llama-32-3B-95-climate-fasrc" # this is the merged model saved locally in the dir "models"


model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map={"": 0},
)

finetuned_model = PeftModel.from_pretrained(model, new_model)
finetuned_model.save_pretrained("adapters/Llama-32-3B-95-climate-fasrc")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = "right"
tokenizer.save_pretrained("adapters/Llama-32-3B-95-climate-fasrc")
merged_model = finetuned_model.merge_and_unload()
merged_model.save_pretrained(local_model)
tokenizer.save_pretrained(local_model)


# 100 % train


base_model = "meta-llama/Llama-3.2-3B-Instruct" # this must leave in HF
new_model = "local/Llama-32-3B-100-climate-fasrc"  # this can leave in huggingface
local_model = "Llama-32-3B-100-climate-fasrc" # this is the merged model saved locally in the dir "models"


model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map={"": 0},
)

finetuned_model = PeftModel.from_pretrained(model, new_model)
finetuned_model.save_pretrained("adapters/Llama-32-3B-100-climate-fasrc")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = "right"
tokenizer.save_pretrained("adapters/Llama-32-3B-100-climate-fasrc")
merged_model = finetuned_model.merge_and_unload()
merged_model.save_pretrained(local_model)
tokenizer.save_pretrained(local_model)
