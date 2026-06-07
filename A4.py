import pandas as pd
import urllib.request
import os
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# LangChain and HuggingFace imports
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==============================================================================
# Part 1: The dataset
# ==============================================================================
print("--- Task 1.1: Downloading and inspecting dataset ---")
# ⚙ Task 1.1. Downloading and inspecting the question answering dataset
# Implementation: download the PubMedQA dataset if it doesn't exist.

url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/refs/heads/master/data/ori_pqal.json"
if not os.path.exists("ori_pqal.json"):
    urllib.request.urlretrieve(url, "ori_pqal.json")

tmp_data = pd.read_json("ori_pqal.json").T
tmp_data = tmp_data[tmp_data.final_decision.isin(["yes", "no"])]

documents = pd.DataFrame({
    "abstract": tmp_data.apply(lambda row: (" ").join(row.CONTEXTS+[row.LONG_ANSWER]), axis=1),
    "year": tmp_data.YEAR
})
questions = pd.DataFrame({
    "question": tmp_data.QUESTION,
    "year": tmp_data.YEAR,
    "gold_label": tmp_data.final_decision,
    "gold_context": tmp_data.LONG_ANSWER,
    "gold_document_id": documents.index
})

# ==============================================================================
# Part 2: Configure your LangChain LM
# ==============================================================================
print("--- Task 2.1: Select a language model ---")
# ⚙ Task 2.1. Select a language model
# Implementation: Using a lightweight instruction-tuned model. 
# For actual deployment or better performance, may replace this with Llama-3 
# (requires HuggingFace token) or Mistral.

hf_pipeline = pipeline(
    "text2text-generation", 
    model="google/flan-t5-base", # Used base for reasonable execution speed
    max_new_tokens=50,
    device=0
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)
'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

# 使用性能更强的因果语言模型，并自动分配到 GPU
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16, # 使用半精度节省显存，加速推理
    device_map="auto"          # 自动调用 Berzelius 上的所有可用 GPU
)

hf_pipeline = pipeline(
    "text-generation", # 注意：T5 是 text2text-generation，而大多数现代开源模型是 text-generation
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    return_full_text=False # 确保 LangChain 只接收生成的答案，而不包含 prompt 本身
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)
'''
# ==============================================================================
# Part 3: Set up the document database
# ==============================================================================
print("--- Task 3.1: Embedding model ---")
#Task 3.1. Embedding model

# employ a HuggingFace sentence transformer ('all-MiniLM-L6-v2') 
# to project document chunks and queries into a shared dense vector space.
# 
#not a standard word-embedding model (like Word2Vec) here
#Standard word embeddings do not inherently capture sequence-level semantics 
#or context. Sentence transformers pool token embeddings specifically to optimize 
#cosine similarity for asymmetric tasks like semantic search.
# 
# the choice of embedding dimensionality affect the pipeline
# Higher dimensions capture more nuanced semantics but increase memory usage 
# and retrieval latency. 'all-MiniLM-L6-v2' (384 dimensions) provides a strong 
# balance between speed and accuracy for PubMed abstracts.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("--- Task 3.2: Chunking ---")
# ⚙ Task 3.2. Chunking
# Implementation: RecursiveCharacterTextSplitter splits documents on logical separators 
# (\n\n, \n, space) to maintain paragraph/sentence coherence.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
metadatas = [{"id": idx} for idx in documents.index]
texts = text_splitter.create_documents(texts=documents.abstract.tolist(), metadatas=metadatas)

print("--- Task 3.3: Define a vector store ---")
# Task 3.3. Define a vector store
# Recap: We instantiate a Chroma vector store using Cosine Similarity. 
# Metadata (doc ID) is injected to preserve the lineage of each chunk.
#
# when the vector database scales to millions of documents
# Exact nearest neighbor (k-NN) becomes computationally infeasible. We would 
# need to transition to Approximate Nearest Neighbor (ANN) algorithms, such 
# as HNSW (Hierarchical Navigable Small World), which trade a small amount of 
# accuracy for massive speedups.
#
# keep the document ID in the metadata
# It is crucial for provenance and evaluation (Task 5.2). It allows us to 
# verify if the retrieved chunk actually originates from the "gold" context.
vector_store = Chroma.from_documents(documents=texts, embedding=embeddings)

# ==============================================================================
# Part 4: Implementing the system
# ==============================================================================
print("--- Task 4.1: Defining the full RAG pipeline ---")
# Task 4.1. Defining the full RAG pipeline (Option B - LCEL)
# LangChain Expression Language (LCEL).
# It uses RunnableParallel to simultaneously retrieve context and pass the user's 
# query into a ChatPromptTemplate, which is then fed to the Generative LLM.
#
# advantages of LCEL over custom wrapper classes (Option A)?
# LCEL offers declarative syntax, native asynchronous support, and stream 
# capabilities out of the box. It makes the data flow explicitly clear.
#
# Q: If the model hallucinates an answer despite the context, what could be the issue?
# A: The issue could be "lost in the middle" (context is too long), poor 
#    retrieval (irrelevant context), or weak instruction-following by the generative 
#    model. Enhancing the prompt with strict negative constraints ("answer ONLY based 
#    on the context") helps mitigate this.

# Retrieve top 1 document
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

template = """Answer the question based ONLY on the following context. 
If the answer is yes, output 'Yes'. If the answer is no, output 'No'.
Context: {context}
Question: {question}
Answer: """

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RunnableParallel captures formatted context, original question, and raw documents for evaluation
setup_and_retrieval = RunnableParallel(
    {"context": retriever | format_docs, "question": RunnablePassthrough(), "raw_docs": retriever}
)

chain = prompt | llm | StrOutputParser()
rag_chain = setup_and_retrieval.assign(answer=chain)

# ==============================================================================
# Part 5: Evaluate RAG on the dataset
# ==============================================================================
print("--- Task 5.1 & 5.2: Evaluation ---")
# Task 5.1. High-level evaluation
# Task 5.2. Detailed inspection
# Recap: We measure end-to-end generation quality using Accuracy (Task 5.1) 
# and intermediate retrieval quality using Recall@1 (Task 5.2).
#
# evaluate the retriever separately from the LLM
# A RAG system has two distinct failure modes: failure to retrieve relevant 
#    information, and failure to generate a correct answer given the information. 
#    Decoupling evaluation isolates the bottleneck.
#

# The baseline LLM relies solely on its parametric memory, which may lead to 
# hallucinations on domain-specific PubMed queries.  
# RAG leverages non-parametric memory, grounding its responses in the retrieved abstract.

def run_evaluation(num_samples=20):
    eval_questions = questions.head(num_samples)
    
    rag_preds, base_preds, gold_labels = [], [], []
    retrieval_hits = 0
    
    # Baseline setup (no retrieval)
    base_template = "Answer the medical question with 'Yes' or 'No'.\nQuestion: {question}\nAnswer:"
    base_prompt = ChatPromptTemplate.from_template(base_template)
    base_chain = base_prompt | llm | StrOutputParser()

    print(f"Evaluating {num_samples} samples...")
    for idx, row in tqdm(eval_questions.iterrows(), total=len(eval_questions)):
        q = row['question']
        gold = row['gold_label'].lower()
        gold_id = row['gold_document_id']
        
        # 1. RAG Prediction & Document Inspection
        rag_res = rag_chain.invoke(q)
        rag_ans = rag_res['answer'].strip().lower()
        rag_pred = "yes" if "yes" in rag_ans else ("no" if "no" in rag_ans else "invalid")
        
        # Check retrieval hit
        retrieved_docs = rag_res['raw_docs']
        retrieved_id = retrieved_docs[0].metadata['id'] if retrieved_docs else None
        if retrieved_id == gold_id:
            retrieval_hits += 1
            
        # 2. Baseline Prediction
        base_res = base_chain.invoke({"question": q})
        base_ans = base_res.strip().lower()
        base_pred = "yes" if "yes" in base_ans else ("no" if "no" in base_ans else "invalid")
        
        rag_preds.append(rag_pred)
        base_preds.append(base_pred)
        gold_labels.append(gold)

    # Filter out invalid generations to calculate accuracy
    valid_rag_idx = [i for i, p in enumerate(rag_preds) if p in ['yes', 'no']]
    valid_base_idx = [i for i, p in enumerate(base_preds) if p in ['yes', 'no']]
    
    rag_acc = accuracy_score([gold_labels[i] for i in valid_rag_idx], [rag_preds[i] for i in valid_rag_idx]) if valid_rag_idx else 0
    base_acc = accuracy_score([gold_labels[i] for i in valid_base_idx], [base_preds[i] for i in valid_base_idx]) if valid_base_idx else 0
    
    print("\n--- Evaluation Results ---")
    print(f"RAG Accuracy (on valid answers): {rag_acc:.2f} (Evaluated on {len(valid_rag_idx)}/{num_samples})")
    print(f"Baseline Accuracy (on valid answers): {base_acc:.2f} (Evaluated on {len(valid_base_idx)}/{num_samples})")
    print(f"Retrieval Accuracy (Recall@1): {retrieval_hits/num_samples:.2f}")

if __name__ == "__main__":
    # Note: For full assignment evaluation, remove or increase the head() limit in run_evaluation
    run_evaluation(20)
