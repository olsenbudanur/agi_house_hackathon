from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
import pymysql
import pymysql.cursors
import fitz  # PyMuPDF
import aiohttp
import json
import os
import logging
from typing import List, Dict

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configure PyMySQL to handle JSON arrays
pymysql.converters.encoders[list] = pymysql.converters.escape_string

# Load environment variables with validation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_HOST = os.getenv("DB_HOST")
DB_PASS = os.getenv("DB_PASS")
DB_USER = os.getenv("DB_USER", "admin")
DB_NAME = os.getenv("DB_NAME", "myvectortable")

# Validate required environment variables
required_vars = ["OPENAI_API_KEY", "DB_HOST", "DB_PASS"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize FastAPI app
app = FastAPI()

async def get_db_connection():
    """Get a database connection with proper error handling"""
    try:
        # Match JavaScript implementation's connection pattern exactly
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            cursorclass=pymysql.cursors.DictCursor,
            # Minimal SSL config like Node.js mysql
            ssl={},
            charset='utf8mb4'
        )
        
        # Test connection immediately like Node.js implementation
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        if not result or not result.get('1'):
            raise pymysql.Error("Connection test failed")
        cursor.close()
        
        print('Connected to SingleStore database successfully')
        return conn
        
    except pymysql.Error as e:
        print(f"Error connecting to database: {str(e)}")
        # Match Node.js error handling pattern
        if isinstance(e, pymysql.OperationalError):
            print("Connection error - check network and credentials")
        if conn:
            try:
                conn.close()
            except:
                pass
        raise
    except Exception as e:
        print(f"Unexpected error while connecting to database: {str(e)}")
        if conn:
            try:
                conn.close()
            except:
                pass
        raise

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
    """Upload embeddings to database with proper connection management"""
    conn = None
    cursor = None
    try:
        conn = await get_db_connection()
        cursor = conn.cursor()
        
        for chunk, embedding in zip(chunks, embeddings):
            query = "INSERT INTO myvectortable (text, vector) VALUES (%s, JSON_ARRAY_PACK(%s));"
            cursor.execute(query, (chunk, json.dumps(embedding)))
        
        conn.commit()
        return {"message": "Upload succeeded"}
        
    except pymysql.Error as db_err:
        print(f"Database error in upload: {str(db_err)}")
        if conn:
            conn.rollback()
        raise
    except Exception as e:
        print(f"Error in upload: {str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.get("/")
async def root():
    return FileResponse("index.html")

@app.get("/resetDatabase")
async def reset_database():
    """Reset the database with proper connection management"""
    conn = None
    cursor = None
    try:
        conn = await get_db_connection()
        if not conn.is_connected():
            raise mysql.connector.Error("Failed to establish database connection")
            
        cursor = conn.cursor()
        cursor.execute("DELETE FROM myvectortable;")
        conn.commit()
        return {"message": "Database reset successful"}
        
    except mysql.connector.Error as db_err:
        print(f"Database error in reset: {str(db_err)}")
        if conn and conn.is_connected():
            conn.rollback()
        raise
    except Exception as e:
        print(f"Error in reset: {str(e)}")
        if conn and conn.is_connected():
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

@app.get("/uploadDataDir")
async def upload_data_dir():
    all_text = await get_all_data()
    chunks = await chunk_text(all_text)
    embeddings = await embed_chunks(chunks)
    await upload_embeddings_to_database(embeddings, chunks)
    return {"message": "Upload succeeded"}

async def query(query_text: str, count: int = 3):
    """Execute a query against the RAG system and return standardized results"""
    conn = None
    cursor = None
    try:
        # Get embedding for query
        embedding_response = await embed(query_text)
        embedding_json = json.dumps(embedding_response['data'][0]['embedding'])
        
        # Connect to database
        conn = await get_db_connection()
        cursor = conn.cursor()
        
        # Match JavaScript implementation's query format exactly
        query = f"""
        SELECT text, dot_product(vector, JSON_ARRAY_PACK(%s)) AS score 
        FROM myvectortable 
        ORDER BY score DESC 
        LIMIT {count};
        """
        cursor.execute(query, (embedding_json,))
        results = cursor.fetchall()
        
        if not results:
            return [{"text": "No relevant information found", "score": 0.0}]
            
        # Standardize response format to match JavaScript implementation exactly
        return [{"text": str(row["text"]), "score": float(row["score"])} for row in results]
        
    except pymysql.Error as db_err:
        print(f"Database error in query function: {str(db_err)}")
        return [{"text": "Database connection error", "score": 0.0}]
    except Exception as e:
        print(f"Error in query function: {str(e)}")
        return [{"text": "Error occurred while processing query", "score": 0.0}]
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

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
