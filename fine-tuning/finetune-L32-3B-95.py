# https://docs.anaconda.com/free/miniconda/
# conda create -n jago python=3.10
#
# conda activate jago
# pip3 install accelerate peft bitsandbytes transformers trl
# pip install huggingface-hub wandb
# #pip install trl==0.11.1 transformers==4.43.1
# https://www.datacamp.com/tutorial/fine-tuning-llama-2
# https://www.linkedin.com/pulse/how-fine-tune-compile-serve-llama2-7b-chat-summarization-limin-ma-potgc/
# huggingface-cli login     [ and pass read token]

# for 1B and 3B 
# https://mlexplained.blog/2023/07/24/fine-tune-llama-2-13b-on-a-single-gpu-on-custom-data/

### To run this script in zsh: nohup python finetune-L2-7B_verified.py > logVerified.txt &!

### nvidia-smi
### gpustat -cp


# module load nvhpc/23.7-fasrc01
# module load cuda/12.4.1-fasrc01
# conda activate jago


from huggingface_hub import HfApi, HfFolder
# HfFolder.save_token("hf_xxx")

api = HfApi()
api.whoami()  # This should print your user info


import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    #TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
import wandb

# Model from Hugging Face hub
base_model = "meta-llama/Llama-3.2-3B-Instruct"

# New instruction dataset
my_dataset = "siacus/climate-finetuning-L3-95"

# Fine-tuned model
new_model = "Llama-32-3B-95-climate-fasrc"
local_model = "local/Llama-32-3B-95-climate-fasrc"

dataset = load_dataset(my_dataset, split={'train': 'train', 'test': 'test'})
compute_dtype = torch.bfloat16 #getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0},
)

model.config.use_cache = False
model.config.pretraining_tp = 1

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = "right"


if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  Allocated Memory: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
        print(f"  Cached Memory: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
else:
    print("CUDA is not available.")


# Define hyperparameters
learning_rate = 2e-4
gradient_accumulation_steps = 1
per_device_train_batch_size = 2
num_train_epochs = 10
max_seq_length = 1024
max_steps = -1
optimizer = "paged_adamw_32bit"
max_grad_norm = 0.3
max_length = 1024

# Initialize W&B with a custom run name
wandb.init(
    project="climate",
    reinit=True,
    name="llama-32-3b-95-climate-fasrc",
    config={
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_seq_length": max_seq_length,
        "max_steps": max_steps,
        "optimizer": optimizer,
        "max_grad_norm": max_grad_norm,
    }
)


peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=64, 
    lora_alpha=32, 
    lora_dropout=0.1,
)
model = get_peft_model(model, peft_config)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

print_trainable_parameters(model)

model.gradient_checkpointing_enable()


# Modify SFTConfig to enable checkpoints
training_params = SFTConfig(
    output_dir="./results95",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    optim=optimizer,
    weight_decay=0.001,
    save_steps=50,  # Save checkpoint every 250 steps (adjust as needed)
    logging_steps=50,
    fp16=False,
    gradient_checkpointing=True,
    bf16=False,
    max_steps=max_steps,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb",
    max_grad_norm=max_grad_norm,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
)



# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_params,
)





# Check for existing checkpoints and resume if found
last_checkpoint = None
if os.path.exists(training_params.output_dir) and os.listdir(training_params.output_dir):
    last_checkpoint = max(
        [os.path.join(training_params.output_dir, d) for d in os.listdir(training_params.output_dir)],
        key=os.path.getctime,
    )
    print(f"Resuming training from checkpoint: {last_checkpoint}")

# Train the model (resume if checkpoint exists)
trainer.train(resume_from_checkpoint=last_checkpoint)

# trainer.train()

trainer.model.save_pretrained(local_model)
trainer.tokenizer.save_pretrained(local_model)

trainer.model.push_to_hub(new_model)
trainer.tokenizer.push_to_hub(new_model)

exit(0)


