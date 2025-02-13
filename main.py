from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import os
import torch
from typing import List

# Initialize FastAPI
app = FastAPI()

# Load AI model (DeepSeek 1.3B)
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with low CPU memory usage
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    device_map="auto" if torch.cuda.is_available() else None,
)

# Initialize HuggingFaceEmbeddings with an explicit model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS index (initially empty)
vector_db = FAISS.from_texts([""], embedding_model)

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

# API endpoint to upload PDFs
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load and process the PDF
        pdf_loader = PyPDFLoader(file_path)
        documents = pdf_loader.load()
        texts = [doc.page_content for doc in documents]

        # Add the text to the FAISS index
        vector_db.add_texts(texts)

        # Clean up the temporary file
        os.remove(file_path)

        return {"message": f"PDF processed! {len(texts)} pages added."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to ask questions
@app.post("/ask/")
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
    uvicorn.run(app, host="0.0.0.0", port=10000)
