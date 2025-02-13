from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import os
import torch

# Initialize FastAPI
app = FastAPI()

# Load AI model (DeepSeek 1.3B)
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with low CPU memory usage (requires accelerate library)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    device_map="auto" if torch.cuda.is_available() else None,
)

# Load PDFs from Google Drive (if running locally)
pdf_folder = "/content/drive/MyDrive/NCTB_PDFs"  # Update this path if needed
all_pdfs = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

# Load and process actual PDF text content
pdf_texts = []
for pdf in all_pdfs:
    pdf_loader = PyPDFLoader(pdf)
    documents = pdf_loader.load()
    pdf_texts.extend([doc.page_content for doc in documents])

# Initialize HuggingFaceEmbeddings with an explicit model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS index with actual PDF text
vector_db = FAISS.from_texts(pdf_texts, embedding_model)

# Create LLM chain
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)
llm_chain = HuggingFacePipeline(pipeline=pipe)

# Set up Kona's QA system
kona_qa = RetrievalQA.from_chain_type(
    llm=llm_chain,
    chain_type="map_reduce",  # Changed from "stuff" to "map_reduce"
    retriever=vector_db.as_retriever(search_kwargs={"k": 5}),  # Retrieve top 5 documents
    return_source_documents=True
)

# Request model for the API
class QuestionRequest(BaseModel):
    question: str

# API endpoint
@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        response = kona_qa({"query": request.question})
        result = response.get('result', 'Sorry, I couldn’t find an answer.').strip()

        # Handle multiple sources
        sources = []
        if response.get('source_documents', []):
            for doc in response['source_documents']:
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                sources.append(f"{source} (Page {page})")

        if sources:
            source_info = "\nSources:\n- " + "\n- ".join(sources)
        else:
            source_info = "\nSource: Unknown"

        return {"answer": result, "sources": sources}

    except Exception as e:
        # Print error to logs or console for debugging purposes
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

