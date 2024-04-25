import pandas as pd
import pyarrow.parquet as pq
from indicnlp.tokenize import indic_tokenize
import re
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from bntransformer import BanglaTokenizer


bntokenizer = BanglaTokenizer() 
table = pq.read_table('banglanews24.parquet')
df = table.to_pandas()
subset_df = df.head(2) 

def process_text(Text):
    Text = re.sub('[^\u0980-\u09FF]', ' ', str(Text))
    return Text

# Tokenize Bengali text in the first column of the DataFrame
tokenized_texts = []
for text in subset_df.iloc[:, 0]:
    if isinstance(text, str) and not text.strip():
        continue
    texts= process_text(text)  
    tokens = bntokenizer.tokenize(texts)

# Create a new DataFrame with tokenized text
df_tokenized = pd.DataFrame(tokens)
print(df_tokenized)

# Step 1: Indexing
index = {}
for idx, row in enumerate(df_tokenized_transposed.values):
    for token in row:
        if token not in index:
            index[token] = []
        index[token].append(idx)



def process_question(question):
    # Tokenize and preprocess the question using the same method as for the dataset
    question_tokens = indic_tokenize.trivial_tokenize(question)
    processed_question = [remove_punctuation(token) for token in question_tokens]
    return processed_question


# Step 3: Answer Retrieval
def retrieve_answers(processed_question):
    matching_rows = set()
    for token in processed_question:
        if token in index:
            matching_rows.update(index[token])
    
    answers = []
    for row_idx in matching_rows:
        # Ensure row index is within bounds
        if row_idx < len(df_tokenized_transposed.index):
            # Retrieve answers from the original DataFrame using loc
            row_answers = df_tokenized_transposed.loc[row_idx].dropna().values.tolist()
            # Filter out unexpected values (empty strings and ',')
            row_answers = [answer for answer in row_answers if answer.strip() and answer != ',']
            answers.extend(row_answers)
    
    return answers



def generate_response(question):
    # Process the question
    processed_question = process_question(question)
    
    # Load pre-trained model and tokenizer
    model_name = "sagorsarker/mbert-bengali-tydiqa-qa"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    # Tokenize user question using BERT tokenizer
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    
    # Perform question answering
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Decode and print the answer
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits)
    answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end+1])
    
    # Retrieve answers from the dataset based on the processed question
    answers = retrieve_answers(processed_question)
    
    if not answers:
        return "Sorry, I couldn't find an answer to your question."
    else:
        # Combine model answer with dataset answers
        answers.append(answer)
        return " ".join(answers)


#user_question = "ঢাকা শহরটির উপকূল কি?"
#response = generate_response(user_question)
#print("Response:", response)