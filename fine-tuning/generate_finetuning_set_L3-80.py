# generates training-set for fine tuning and testing
# pip install scikit-learn

import json
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

max_abs_size = 3000

df = pd.read_csv("allcoded.csv", sep = '\t', dtype=str)
df = df.fillna("")

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

# this splits 20% train / 20% test
train_set, test_set = train_test_split(df, test_size=0.20, random_state=123)

train = train_set
test = test_set 


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




def create_QueryAnswer(title, abstract, ansQ1, ansQ2, ansQ3, ansQ4, ansQ5, ansQ6):
    if len(abstract) > max_abs_size:
        abstract = abstract[:max_abs_size]
    tmp = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{instructionSystem}<|eot_id|><|start_header_id|>user<|end_header_id|>{instruction}.\nTitle:"{title}."\nAbstract:"{abstract}."\n{questions}<|eot_id|><|start_header_id|>assistant<|end_header_id|>Answer:\nQ1 = [{ansQ1}];\nQ2 = [{ansQ2}];\nQ3 = [{ansQ3}];\nQ4 = [{ansQ4}];\nQ5 = [{ansQ5}];\nQ6 = [{ansQ6}]."""
    return tmp

def create_Query(title, abstract):
    if len(abstract) > max_abs_size:
        abstract = abstract[:max_abs_size]
    tmp = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{instructionSystem}<|eot_id|><|start_header_id|>user<|end_header_id|>{instruction}.\nTitle:"{title}."\nAbstract:"{abstract}."\n{questions}<|eot_id|><|start_header_id|>assistant<|end_header_id|>Answer:\n"""
    return tmp


# Apply the transformation
QA = train.apply(lambda row: create_QueryAnswer(row['title'], row['abstract'], row['Q1'], row['Q2'], row['Q3'], row['Q4'], row['Q5'], row['Q6']), axis=1)
QA = QA.to_frame("text")
QA['id'] = train['id']
QA['doi'] = train['doi']
QA['title'] = train['title']
QA['abstract'] = train['abstract']

trainQueries = QA
trainQueries.reset_index(drop=True, inplace=True)

Q = test.apply(lambda row: create_Query(row['title'], row['abstract']), axis=1)
Q = Q.to_frame("text")
Q['id'] = test['id']
Q['doi'] = test['doi']
Q['title'] = test['title']
Q['abstract'] = test['abstract']

testQueries = Q
testQueries.reset_index(drop=True, inplace=True)


dataset_train = Dataset.from_pandas(trainQueries)
dataset_test = Dataset.from_pandas(testQueries)

dataset_dict = DatasetDict({
    "train": dataset_train,
    "test" : dataset_test
})
# Now 'dataset' is a Hugging Face Dataset
from huggingface_hub import HfApi, HfFolder

# Your Hugging Face token
#hf_token = "hf_xxx"

# Save the token (this will save the token to the HfFolder's location, typically ~/.huggingface)
#HfFolder.save_token(hf_token)

# Optionally, you can also set the token directly for the current script execution
#HfApi().set_access_token(hf_token)
api = HfApi()
api.whoami()  # This should print your user info


dataset_dict.push_to_hub("siacus/climate-finetuning-L3-80")




