# build llama-cpp-python as in "various_cluster_intructions.txt"
# the do the followin
# 
# you can make it a sbatch script
#
# salloc -p gpu_test --gres=gpu:1 --mem=40G -N 1 -t 120
# module load nvhpc/23.7-fasrc01
# module load cuda/12.2.0-fasrc01 
# module load gcc/12.2.0-fasrc01
# module load cmake
# conda activate jago


from llama_cpp import Llama
# We'll use pprint to more clearly look at the output
from pprint import pprint
import re
import pandas as pd
from tqdm import tqdm
import time
import os

# Create an instance of Llama to load the model
# model_path - The model we want to load
llm = Llama(
    model_path="../gguf/llama32-3B-100-climate-fasrc-Q4_K_M.gguf",
    n_ctx = 2048,
    n_gpu_layers = -1,
    verbose=False
)

print("Model loaded")


max_abs_size = 3000

train = pd.read_csv("wos_migration_mapped.csv", dtype=str)
train = train.fillna("")

# Q1: Is the article about human migration and mobility?
# candidate_labels = ['human migration mobility']
catDictQ1 = {'no': 0, 'yes': 1}
hintQ1 = result_string = ", ".join(f"{v} = '{k}'" for k, v in catDictQ1.items())

# Q2: What drivers of migration and mobility does the article study?
# multiple choice
drivers = ['other', 'climate', 'environment', 'economic', 'conflict', 'political', 'social', 'family']
driversCodes =  list(range(len(drivers)))
catDictQ2 = dict(zip(drivers, driversCodes))
hintQ2 = result_string = ", ".join(f"{v} = '{k}'" for k, v in catDictQ2.items())


# Q3 What is the overall sentiment of the article?
sentiments = ['neutral', 'negative', 'positive']
sentimentsCodes =  list(range(len(sentiments)))
catDictQ3 = dict(zip(sentiments, sentimentsCodes))
hintQ3 = result_string = ", ".join(f"{v} = '{k}'" for k, v in catDictQ3.items())

# Q4 What type of climate/environmental hazards, if any, is discussed in the article?
# multiple choice
file_path = 'unique_hazards.txt'
with open(file_path, 'r') as file:
    hazards = [line.strip() for line in file]

hazards = ['none'] + hazards + ['other']
hazardsCodes =  list(range(len(hazards)))
catDictQ4 = dict(zip(hazards, hazardsCodes))
hintQ4 = result_string = ", ".join(f"{v} = '{k}'" for k, v in catDictQ4.items())

# Q5 What discipline does the article belong to?
disciplines = ['social science','natural science','arts and humanities', 'multidiscipline and interdiscipline']
disciplines = ['other'] + disciplines
disciplinesCodes =  list(range(len(disciplines)))
catDictQ5 = dict(zip(disciplines, disciplinesCodes))
hintQ5 = result_string = ", ".join(f"{v} = '{k}'" for k, v in catDictQ5.items())

# Q6 Does the article apply quantitative or qualitative method?
methods = ['no','yes']
methodsCodes =  list(range(len(methods)))
catDictQ6 = dict(zip(methods, methodsCodes))
hintQ6 = result_string = ", ".join(f"{v} = '{k}'" for k, v in catDictQ6.items())

instruction = f"""Here are the title and abstract of a scientific paper probably related to climate change, human mobility and migration."""

instructionSystem = """You are an expert on climate change, human mobility and migration. Answer the question truthfully."""



questions = f"""I have six questions you need to answer.
Q1: Is this article is about 'human migration mobility': 1 = yes; 0 = no. Answer with a number. Do not give a textual explanation.

Q2: What drivers of migration and mobility does the article study? {hintQ2}. Answer with a number or a sequence of numbers separated by commas. Do not give a textual explanation. 

Q3: What is the overall sentiment of the article?:{hintQ3}. Answer with a number. Do not give a textual explanation.

Q4: What type of climate or environmental keywords are explicitly mentioned?:  {hintQ4}. Answer with a number or a sequence of numbers separated by commas. Do not give a textual explanation. 

Q5: What discipline does the article belong to?: {hintQ5}. Answer with a number. Do not give a textual explanation.

Q6: Does the article apply quantitative and/or statistical analysis?: {hintQ6}. Answer with a number. Do not give a textual explanation.

Use the following template: 

Q1 = [];
Q2 = [];
Q3 = [];
Q4 = [];
Q5 = [];
Q6 = []."""


def create_Query(title, abstract):
    if len(abstract) > max_abs_size:
        abstract = abstract[:max_abs_size]
    tmp = f"""<|start_header_id|>system<|end_header_id|>{instructionSystem}<|eot_id|><|start_header_id|>user<|end_header_id|>{instruction}.\nTitle:"{title}."\nAbstract:"{abstract}."\n{questions}<|eot_id|><|start_header_id|>assistant<|end_header_id|>Answer:"""
    return tmp

import pandas as pd
import re

def extract_answers(text, marker="Answer:"):
    start_index = text.find(marker)
    if start_index != -1:
        relevant_text = text[start_index + len(marker):]
    else:
        return pd.DataFrame(columns=["Question", "Answer"])
    # Regular expression to match Q1 through Q6 strictly
    pattern = r"(Q[1-6]) = (\[[^\]]*\])"
    matches = re.findall(pattern, relevant_text)
    # Create a DataFrame from matches
    df = pd.DataFrame(matches, columns=["Question", "Answer"])
    # Clean up the answers by removing brackets and ensuring consistent formatting
    df["Answer"] = df["Answer"].apply(lambda x: re.sub(r"[\[\],;]", " ", x).strip())
    # Remove duplicates, keeping the first occurrence only
    df = df.drop_duplicates(subset=["Question"], keep="first")
    # Ensure all Q1 to Q6 are present
    expected_questions = [f"Q{i}" for i in range(1, 7)]
    missing_questions = set(expected_questions) - set(df["Question"])
    # Add missing questions with empty answers
    missing_df = pd.DataFrame({"Question": list(missing_questions), "Answer": [""] * len(missing_questions)})
    # Combine existing and missing rows, then sort
    df = pd.concat([df, missing_df], ignore_index=True)
    df = df.sort_values("Question").reset_index(drop=True)
    return df





def askLLM(query, marker="Answer:"):
    start_index = query.find(marker)
    query = query[:start_index+len(marker)]
    output = llm(
        query,  # Prompt
        max_tokens=100,  # Generate up to 128 tokens
        echo=True,  # Echo the prompt back in the output
        temperature = 0 #0.01
    )
    answer = output['choices'][0]['text']
    return extract_answers(answer)





df = train


output_file = "classifications-wos-100.csv"
if os.path.exists(output_file):
    os.remove(output_file)

n = df.shape[0]
cnt = 0
start_time = time.time()
for index, row in tqdm(df.iterrows(), total=n, desc="Processing rows"):
    query = create_Query(row['Article Title'], row['Abstract'])
    tmp = askLLM(query)
    transformed = tmp.set_index('Question').T
    transformed['id'] = row['id']
    transformed.to_csv(output_file, mode='a', header=not index, index=False)

end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Time spent running the code: {elapsed_time:.2f} seconds")

exit(0)
