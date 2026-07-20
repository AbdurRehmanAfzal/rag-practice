from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from openai import OpenAI
import uuid
import json
import chromadb
from typing import Optional

load_dotenv()
app = FastAPI()

# ============================================
# STARTUP: Model aur data load karna (ek baar)
# ============================================

print("Model load ho raha hai...")
model = SentenceTransformer('all-MiniLM-L6-v2')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================
# NAYA: Ek "Tool/Function" Jo LLM Call Kar Sakta Hai
# ============================================

# Simple project database (real mein yeh Postgres/JSON file se aata)
projects_database = {
    "b1properties": {
        "name": "B1 Properties",
        "url": "b1properties.ae",
        "tech_stack": "Django 5.2, DRF, Next.js 16, PostgreSQL, AWS S3",
        "status": "Live in production",
        "key_metric": "50+ REST API endpoints, zero downtime since launch"
    },
    "workfly": {
        "name": "WorkFly / TimeKeepers",
        "url": "workfly.app",
        "tech_stack": "Django 4.0, Angular 15, Django Channels, Celery, AWS ECS",
        "status": "Live in production",
        "key_metric": "Reduced workforce admin overhead by 50%"
    },
    "palletfly": {
        "name": "Palletfly",
        "url": "dbr.palletfly.com",
        "tech_stack": "Django 3.2, Django Channels, Huey, Redis, PostgreSQL",
        "status": "Live in production",
        "key_metric": "40,000+ lines of Python, 91+ ORM models"
    }
}

def get_full_project_details(project_key: str):
    """Yeh actual Python function hai jo LLM 'call' karega"""
    project = projects_database.get(project_key.lower())
    if project:
        return project
    else:
        return {"error": "Project not found"}


# LLM ko batane ke liye "tool definition" — yeh JSON schema hai
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_full_project_details",
            "description": "Get complete structured details (tech stack, status, metrics) about one of Abdur Rehman's specific projects. Use this when the user asks for deep/specific details about a named project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_key": {
                        "type": "string",
                        "description": "The project identifier, one of: b1properties, workfly, palletfly"
                    }
                },
                "required": ["project_key"]
            }
        }
    }
]

# ============================================
# NAYA: Lead Ka "Schema" Define Karna
# ============================================

class LeadInfo(BaseModel):
    name: Optional[str] = Field(description="Person's name, agar mention ki gayi ho")
    phone: Optional[str] = Field(description="Phone number, agar diya gaya ho")
    email: Optional[str] = Field(description="Email address, agar diya gaya ho")
    property_interest: Optional[str] = Field(description="Kis type ki property mein interest hai")
    budget: Optional[str] = Field(description="Budget range agar mention hua ho")
    is_lead: bool = Field(description="True agar user ne apna koi contact info diya ho, warna False")


def extract_lead_info(conversation_text: str):
    """User ke message se lead information nikalta hai, structured format mein"""

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract lead information from the user's message. If no contact info is present, set is_lead to false."},
            {"role": "user", "content": conversation_text}
        ],
        response_format=LeadInfo   # Yahan hum schema batate hain
    )

    return response.choices[0].message.parsed

# ============================================
# NAYA: ChromaDB Setup (manual cosine similarity ki jagah)
# ============================================

with open("knowledge_base.txt", "r", encoding="utf-8") as file:
    content = file.read()

chunks = content.split("\n\n")
cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip() != ""]

# ChromaDB client — disk pe "./chroma_data" folder mein data persist karega
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="portfolio_knowledge")

# Sirf pehli baar (ya jab knowledge_base.txt update ho) embeddings banao aur store karo
if collection.count() != len(cleaned_chunks):
    print("Embeddings ban rahi hain aur ChromaDB mein store ho rahi hain...")

    # Agar purana (mismatched) data pada hai, pehle usay saaf karte hain
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    chunk_embeddings = model.encode(cleaned_chunks)
    collection.add(
        documents=cleaned_chunks,
        ids=[f"chunk_{i}" for i in range(len(cleaned_chunks))],
        embeddings=chunk_embeddings.tolist()
    )
    print("Store ho gaya!")
else:
    print(f"ChromaDB mein pehle se {collection.count()} chunks maujood hain — dobara banane ki zaroorat nahi")

print("RAG system tayar hai!")

# ============================================
# NAYA: Conversation Memory Store
# ============================================

# Yeh ek "dictionary" hai jo har session ki history store karegi
# Format: { "session_id_1": [messages...], "session_id_2": [messages...] }
conversation_store = {}

SYSTEM_PROMPT = """Aap Abdur Rehman Afzal ke portfolio ke AI assistant hain. 
Aap unke professional experience, skills, projects ke sawalon ka jawab dete hain.

Agar koi apna interest ya contact info share kare (jaise job opportunity, hiring, 
ya collaboration ke liye), unhe warmly acknowledge karein aur bataiye ke Abdur Rehman 
jald unse contact karenge.

Sirf tab politely decline karein jab sawal bilkul unrelated ho (jaise weather, 
general knowledge, ya kisi aur topic pe), aur bataiye ke aap sirf Abdur Rehman ke 
professional background mein madad kar sakte hain."""

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

    # Step 3 (UPDATED): Retrieval — ab ChromaDB se
    question_embedding = model.encode([question])

    results = collection.query(
        query_embeddings=[question_embedding[0].tolist()],
        n_results=3
    )

    top_chunks = results['documents'][0]
    context = "\n".join(top_chunks)

    # Step 4: Naya "user" message banate hain, jisme retrieval context bhi hai
    user_message_with_context = f"""Yahan kuch relevant information hai:

{context}

Sawal: {question}"""

    # Step 5: Poori messages list banana — system + purani history + naya sawal
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)   # purani conversation add karna
    messages.append({"role": "user", "content": user_message_with_context})

    # Step 6: OpenAI ko messages + tools bhejna
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools  # naya parameter — LLM ko available tools batate hain
    )

    response_message = response.choices[0].message

    # Step 6b: Check karna — kya LLM ne function call karna chaha?
    if response_message.tool_calls:
        # LLM ne function call maanga hai
        tool_call = response_message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        print(f"LLM ne function call kiya: {function_name}({function_args})")

        # Actual function ko chalana
        if function_name == "get_full_project_details":
            function_result = get_full_project_details(function_args.get("project_key"))

        # LLM ka function-call message, aur function ka result, dono conversation mein add karna
        messages.append(response_message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(function_result)
        })

        # Dobara LLM ko bhejna — is baar result ke sath, taake natural jawab bane
        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        answer = second_response.choices[0].message.content
    else:
        # Normal jawab tha, function call nahi hua
        answer = response_message.content

    # Step 7: Is exchange ko history mein save karna (agli baar ke liye)
    history.append({"role": "user", "content": question})   # original sawal (bina context ke) save karte hain
    history.append({"role": "assistant", "content": answer})
    conversation_store[session_id] = history

    # Step 8: Har message se lead info nikalne ki koshish karna
    lead_info = extract_lead_info(question)

    if lead_info.is_lead:
        print(f"🎯 NAYA LEAD MILA: {lead_info}")
        # Agle step mein yahan hum n8n ko webhook call karenge!

    return {
        "question": question,
        "answer": answer,
        "sources": top_chunks,
        "session_id": session_id,   # Frontend ko wapis bhejna, taake agli request mein use ho
        "lead_captured": lead_info.is_lead
    }

# ============================================
# Static Files Serve Karna
# ============================================

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")