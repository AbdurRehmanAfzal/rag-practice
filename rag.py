from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI
import uuid

load_dotenv()
app = FastAPI()

# ============================================
# STARTUP: Model aur data load karna (ek baar)
# ============================================

print("Model load ho raha hai...")
model = SentenceTransformer('all-MiniLM-L6-v2')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("knowledge_base.txt", "r", encoding="utf-8") as file:
    content = file.read()

chunks = content.split("\n\n")
cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip() != ""]
chunk_embeddings = model.encode(cleaned_chunks)

print("RAG system tayar hai!")

# ============================================
# NAYA: Conversation Memory Store
# ============================================

# Yeh ek "dictionary" hai jo har session ki history store karegi
# Format: { "session_id_1": [messages...], "session_id_2": [messages...] }
conversation_store = {}

SYSTEM_PROMPT = """Aap Abdur Rehman Afzal ke portfolio ke AI assistant hain. 
Aap sirf unke professional experience, skills, aur projects ke baare mein sawalon ka jawab dete hain, 
di gayi information ke bunyad par. Agar koi sawal iske alawa ho, politely bataiye ke aap sirf 
Abdur Rehman ke professional background ke baare mein madad kar sakte hain."""

# ============================================
# REQUEST BODY
# ============================================

class QuestionRequest(BaseModel):
    question: str
    session_id: str | None = None   # Optional — agar na aaye, naya banayenge

# ============================================
# API ENDPOINT
# ============================================

@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question

    # Step 1: Session ID handle karna
    session_id = request.session_id
    if session_id is None or session_id not in conversation_store:
        session_id = str(uuid.uuid4())   # Naya unique ID banate hain
        conversation_store[session_id] = []  # Khaali history se shuru

    # Step 2: Is session ki purani history nikalna
    history = conversation_store[session_id]

    # Step 3: Retrieval — sawal ka embedding aur relevant chunks dhoondna
    question_embedding = model.encode([question])
    similarities = cosine_similarity(question_embedding, chunk_embeddings)
    top_n = 3
    top_indices = np.argsort(similarities[0])[::-1][:top_n]
    top_chunks = [cleaned_chunks[i] for i in top_indices]
    context = "\n".join(top_chunks)

    # Step 4: Naya "user" message banate hain, jisme retrieval context bhi hai
    user_message_with_context = f"""Yahan kuch relevant information hai:

{context}

Sawal: {question}"""

    # Step 5: Poori messages list banana — system + purani history + naya sawal
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)   # purani conversation add karna
    messages.append({"role": "user", "content": user_message_with_context})

    # Step 6: OpenAI ko poori conversation bhejna
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    answer = response.choices[0].message.content

    # Step 7: Is exchange ko history mein save karna (agli baar ke liye)
    history.append({"role": "user", "content": question})   # original sawal (bina context ke) save karte hain
    history.append({"role": "assistant", "content": answer})
    conversation_store[session_id] = history

    return {
        "question": question,
        "answer": answer,
        "sources": top_chunks,
        "session_id": session_id   # Frontend ko wapis bhejna, taake agli request mein use ho
    }

# ============================================
# Static Files Serve Karna
# ============================================

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")