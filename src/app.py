import os
os.environ["TOKENIERS_PARALLELISM"] = "false"

os.environ["LANGSMITH_API_KEY"] = ""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes

from src.base.llm_model import get_hf_llm
from src.rag.main import build_rag_chain, InputQA, OutputQA

llm = get_hf_llm(temperature=0.9)
genai_docs = "/Users/genson1808/workspace/ai/chat_rag/data_source/docs"

genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")

app = FastAPI(
    title="Langchain Serve",
    description="GenAI API",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/check")
async def check():
    return {"status": "OK"}

@app.post("/generative_ai", response_model=OutputQA)
async def generative_ai(inputs: InputQA):
    answer = genai_chain.invoke(inputs.question)
    return {"answer": answer}

# Check the spelling of the path here
add_routes(app, genai_chain, playground_type="default", path="/generative_ai")
