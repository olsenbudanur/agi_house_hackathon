from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
import mysql.connector
import fitz  # PyMuPDF
import aiohttp
import json
import os
from typing import List, Dict

# Environment variables
OPENAI_API_KEY = "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"
DB_HOST = "svc-17a93717-cca1-4255-b279-4e2be1f55cab-dml.aws-virginia-8.svc.singlestore.com"
DB_PASS = "8GZttiDd80q4TUfu5KAuosRZFmVzxXHF"
DB_USER = "admin"
DB_NAME = "myvectortable"

app = FastAPI()

async def get_db_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )

async def pdf_to_text(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

async def get_all_data() -> str:
    all_text = ""
    for file in os.listdir("./data"):
        if file.endswith(".pdf"):
            all_text += await pdf_to_text(f"./data/{file}")
    return all_text

async def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    sentences = text.split(".")
    chunks = []
    chunk = ""
    for sentence in sentences:
        chunk += sentence + "."
        if len(chunk) > chunk_size:
            chunks.append(chunk)
            chunk = ""
    return chunks

async def embed(text: str) -> Dict:
    async with aiohttp.ClientSession() as session:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {OPENAI_API_KEY}'
        }
        async with session.post(
            'https://api.openai.com/v1/embeddings',
            headers=headers,
            json={'model': "text-embedding-ada-002", 'input': text}
        ) as response:
            if response.status != 200:
                raise Exception(f"HTTP error! status: {response.status}")
            return await response.json()

async def embed_chunks(chunks: List[str]) -> List:
    embeddings = []
    for chunk in chunks:
        response = await embed(chunk)
        embeddings.append(response['data'][0]['embedding'])
    return embeddings

async def upload_embeddings_to_database(embeddings: List, chunks: List[str]):
    conn = await get_db_connection()
    cursor = conn.cursor()
    
    for chunk, embedding in zip(chunks, embeddings):
        query = "INSERT INTO myvectortable (text, vector) VALUES (%s, JSON_ARRAY_PACK(%s));"
        cursor.execute(query, (chunk, json.dumps(embedding)))
    
    conn.commit()
    cursor.close()
    conn.close()

@app.get("/")
async def root():
    return FileResponse("index.html")

@app.get("/resetDatabase")
async def reset_database():
    conn = await get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM myvectortable;")
    conn.commit()
    cursor.close()
    conn.close()
    return {"message": "Database reset successful"}

@app.get("/uploadDataDir")
async def upload_data_dir():
    all_text = await get_all_data()
    chunks = await chunk_text(all_text)
    embeddings = await embed_chunks(chunks)
    await upload_embeddings_to_database(embeddings, chunks)
    return {"message": "Upload succeeded"}

async def query(query_text: str):
    """Execute a query against the RAG system"""
    try:
        embedding_response = await embed(query_text)
        
        conn = await get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        query = f"""
        SELECT text, dot_product(vector, JSON_ARRAY_PACK(%s)) AS score 
        FROM myvectortable 
        ORDER BY score DESC 
        LIMIT 3;
        """
        cursor.execute(query, (json.dumps(embedding_response['data'][0]['embedding']),))
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        if not results:
            return [{"text": "No relevant information found", "score": 0}]
        return results
    except Exception as e:
        print(f"Error in query function: {str(e)}")
        return [{"text": "Error occurred while processing query", "score": 0}]

async def chat(messages: List[Dict[str, str]]):
    """Execute a chat interaction with the RAG system"""
    try:
        if len(messages) > 5:
            messages = messages[-5:]
        
        last_message = messages[-1]["content"]
        last_message_info = await query(last_message)
        context = last_message_info[0]["text"] if last_message_info else ""
        messages[-1]["content"] = f"{last_message} - Use this info to answer the question: {context}"
        
        async with aiohttp.ClientSession() as session:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {OPENAI_API_KEY}'
            }
            async with session.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": messages,
                    "temperature": 0.7
                }
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP error! status: {response.status}")
                response_data = await response.json()
                return response_data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error in chat function: {str(e)}")
        return "I apologize, but I encountered an error while processing your request."

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0000", port=3000)
