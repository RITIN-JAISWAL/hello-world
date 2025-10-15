# RAG + Prompt Engineering for OCR Text Classification with GPT-4 Mini (Azure OpenAI)

import pandas as pd
import numpy as np
import json
import re
from sklearn.metrics import classification_report
from tqdm import tqdm
from openai import AzureOpenAI

# === CONFIGURATION ===
endpoint = "https://product-coding.openai.azure.com/"
deployment = "gpt-40-mini"
subscription_key = "YOUR_API_KEY"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_key=subscription_key,
    api_version=api_version,
    azure_endpoint=endpoint
)

EMBEDDING_MODEL = "text-embedding-ada-002"
TOP_K = 5

# === LOAD DATA ===
train_df = pd.read_csv("Kantar_train.csv")
test_df = pd.read_csv("Kantar_test.csv")

# === CLEAN & PREPARE TRAINING DATA ===
train_df["clean_description"] = train_df["clean_description"].fillna("")
train_df["Quantity"] = train_df["Quantity"].fillna("N/A").astype(str)
train_df["Unit"] = train_df["Unit"].fillna("N/A")

# === EMBEDDING TRAINING TEXT ===
def get_embeddings(texts):
    from openai import AzureOpenAI
    import time
    embeddings = []
    client = AzureOpenAI(
        api_key=subscription_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        response = client.embeddings.create(
            input=batch,
            model=EMBEDDING_MODEL
        )
        embeddings.extend([d.embedding for d in response.data])
        time.sleep(1)
    return np.array(embeddings).astype("float32")

train_embeddings = get_embeddings(train_df["clean_description"].tolist())

# === BUILD FAISS INDEX ===
import faiss
index = faiss.IndexFlatL2(train_embeddings.shape[1])
index.add(train_embeddings)

# === RAG RETRIEVAL ===
def retrieve_similar_examples(query, k=TOP_K):
    query_emb = get_embeddings([query])[0]
    D, I = index.search(np.array([query_emb]), k)
    examples = []
    for idx in I[0]:
        row = train_df.iloc[idx]
        example = f"OCR: {row['clean_description']}\n→ Sector: {row['Sector']}; Categoría: {row['Categoría']}; Marca: {row['Marca']}; Size: {row['Quantity']}; Unit: {row['Unit']}"
        examples.append(example)
    return "\n\n".join(examples)

# === STATIC CONTEXT EXAMPLES FOR BOOSTING ACCURACY ===
def get_static_examples(n=10):
    static_examples = []
    sample = test_df.dropna(subset=["ocr_text"]).head(n)
    for _, row in sample.iterrows():
        ex = f"OCR: {row['ocr_text']}\n→ Sector: {row['Sector']}; Categoría: {row['Categoría']}; Marca: {row['Marca']}; Size: {row['OCR_Size'] or 'N/A'}; Unit: {row['OCR_Measure'] or 'N/A'}"
        static_examples.append(ex)
    return "\n\n".join(static_examples)

static_examples_block = get_static_examples(10)

# === PROMPT BUILDER ===
def build_prompt(context_examples, new_ocr):
    return f"""You are an expert assistant that extracts product attributes from OCR text.
Use the examples below to identify Sector, Categoría, Marca, Size and Unit.
If any information is missing, return 'N/A'.

{context_examples}

{static_examples_block}

Now classify this new product:
OCR: {new_ocr}

Your output:
Sector:
Categoría:
Marca:
Size:
Unit:
"""

# === PARSE MODEL OUTPUT ===
def parse_output(output):
    result = {"Sector": "", "Categoría": "", "Marca": "", "Size": "", "Unit": ""}
    for key in result:
        match = re.search(fr"{key}:[ \t]*(.+)", output, re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()
        else:
            result[key] = "N/A"
    return result

# === GPT CALL ===
def gpt4_classify(ocr_text):
    examples = retrieve_similar_examples(ocr_text)
    prompt = build_prompt(examples, ocr_text)
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a structured text classifier."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=1024
    )
    return parse_output(response.choices[0].message.content)

# === EVALUATION ===
y_true = test_df[["Sector", "Categoría", "Marca", "OCR_Size", "OCR_Measure"]].fillna("N/A")
y_pred = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    ocr = row["ocr_text"]
    if not isinstance(ocr, str):
        y_pred.append({"Sector": "N/A", "Categoría": "N/A", "Marca": "N/A", "Size": "N/A", "Unit": "N/A"})
    else:
        y_pred.append(gpt4_classify(ocr))

# === F1 SCORE ===
y_pred_df = pd.DataFrame(y_pred)
for col in ["Sector", "Categoría", "Marca", "Size", "Unit"]:
    print(f"\nF1 Report for {col}:")
    print(classification_report(y_true[col], y_pred_df[col], zero_division=0))
