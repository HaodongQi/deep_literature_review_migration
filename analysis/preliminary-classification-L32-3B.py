
# conda install llama-cpp-python
# https://www.gdcorner.com/blog/2024/06/12/NoBSIntroToLLMs-1-GettingStarted.html
from llama_cpp import Llama
# We'll use pprint to more clearly look at the output
from pprint import pprint
from datasets import load_dataset
import re
import pandas as pd
from tqdm import tqdm
import time
import os
# from transformers import AutoModelForCausalLM
from huggingface_hub import login
from huggingface_hub import hf_hub_download

###########################################
# Create an instance of Llama to load the model
    # download the model from hf
llm = hf_hub_download(
	repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
	filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
)
model_path=f"{llm}"
    # load model from cache
llm = Llama(model_path=model_path,
    n_ctx=2048,  # Context window size
    n_threads=12,  # Number of CPU threads to use
    n_gpu_layers=20  # Number of layers to offload to GPU (if GPU is available)
)
    # Test the model
        # Define a prompt
prompt = "what is re-supervising large language model"
        # Generate text
output = llm(prompt, max_tokens=50,  echo=True)
        # Print the generated text
print(output["choices"][0]["text"])


###########################################
print("Model loaded")
cap_dataset = "siacus/climate-verification-L3"
dataset = load_dataset(cap_dataset, split={'verification': 'verification'})


def get_response(answer):
    match = re.search(r'Answer =.*?(\d+)', answer)
    if match:
        return int(match.group(1))
    else:
        return(0)

def get_response_multi(answer):
    match = re.search(r"Answer\s*=\s*([\d,\s]+)", answer)
    if match:
        # Split the matched numbers by commas and convert them to a set to remove duplicates
        numbers = match.group(1)
        numbers = re.findall(r"\d+", numbers)
        numbers = [num for num in numbers if num.strip() != '']
        unique_numbers = list(set(map(int, numbers)))  # Convert to integers
        # Optional: Sort the numbers
        unique_numbers.sort()
        result = ",".join(str(num) for num in unique_numbers)
    else:
        result = ""
    return(result)


df = pd.DataFrame(dataset['verification'])

results_file = 'classification_verification-L32-3B.csv'
if os.path.exists(results_file):
    os.remove(results_file)

n = df.shape[0]
cnt = 0
results = []
start_time = time.time()
for index, row in tqdm(df.iterrows(), total=n, desc="Processing papers"):
    id = row['id']
    Q1 = row['Q1']
    Q2 = row['Q2']
    Q3 = row['Q3']
    Q4 = row['Q4']
    Q5 = row['Q5']
    Q6 = row['Q6']
    outQ1 = llm(Q1, max_tokens=50, echo=True,  temperature = 0.01)
    outQ2 = llm(Q2, max_tokens=50, echo=True,  temperature = 0.01)
    outQ3 = llm(Q3, max_tokens=50, echo=True,  temperature = 0.01)
    outQ4 = llm(Q4, max_tokens=50, echo=True,  temperature = 0.01)
    outQ5 = llm(Q5, max_tokens=50, echo=True,  temperature = 0.01)
    outQ6 = llm(Q6, max_tokens=50, echo=True,  temperature = 0.01)
    ansQ1 = outQ1['choices'][0]['text']
    ansQ2 = outQ2['choices'][0]['text']
    ansQ3 = outQ3['choices'][0]['text']
    ansQ4 = outQ4['choices'][0]['text']
    ansQ5 = outQ6['choices'][0]['text']
    ansQ6 = outQ6['choices'][0]['text']
    respQ1 = get_response(ansQ1)
    respQ2 = get_response_multi(ansQ2)
    respQ3 = get_response(ansQ3)
    respQ4 = get_response_multi(ansQ4)
    respQ5 = get_response(ansQ5)
    respQ6 = get_response(ansQ6)
    result_row = pd.DataFrame([{
        'id': id,
        'Q1': respQ1,
        'Q2': respQ2,
        'Q3': respQ3,
        'Q4': respQ4,
        'Q5': respQ5,
        'Q6': respQ6
    }])
    # results.append({
    #     'id': id,
    #     'Q1': respQ1,
    #     'Q2': respQ2,
    #     'Q3': respQ3,
    #     'Q4': respQ4,
    #     'Q5': respQ5,
    #     'Q6': respQ6
    # })
    print(f"Processed {len(results)} out of {n}")
    write_header = not os.path.exists(results_file)
    result_row.to_csv(results_file, mode='a', index=False, header=write_header)

end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Time spent running the code: {elapsed_time:.2f} seconds")


# results_df = pd.DataFrame(results)
# results_df.to_csv('classification_verification-L32-3B.csv', index=False)

exit(0)


