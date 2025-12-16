"""
FastAPI backend service for MultiAgenticRAG
Supports: WebSocket streaming, PDF upload, document management
"""
import asyncio
import os
import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
from pydantic import BaseModel

from main_graph.graph_state import InputState
from main_graph.graph_builder import graph
from utils.utils import new_uuid

# Initialize FastAPI app
app = FastAPI(title="MultiAgenticRAG API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PAPERS_DIR = Path(__file__).parent / "papers"
PAPERS_DIR.mkdir(exist_ok=True)


class QueryRequest(BaseModel):
    """Request model for chat queries"""
    query: str
    thread_id: Optional[str] = None


class DocumentInfo(BaseModel):
    """Document information model"""
    filename: str
    size: int
    upload_time: str


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "MultiAgenticRAG"}


@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    """List all uploaded PDF documents"""
    documents = []
    if PAPERS_DIR.exists():
        for pdf_file in PAPERS_DIR.glob("*.pdf"):
            documents.append(
                DocumentInfo(
                    filename=pdf_file.name,
                    size=pdf_file.stat().st_size,
                    upload_time=pdf_file.stat().st_mtime.__str__(),
                )
            )
    return documents


@app.get("/documents/{filename}")
async def download_document(filename: str):
    """Download a specific PDF document"""
    file_path = PAPERS_DIR / filename
    
    # Security check: prevent path traversal
    if not file_path.exists() or not str(file_path).startswith(str(PAPERS_DIR)):
        raise HTTPException(status_code=404, detail="Document not found")
    
    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=filename
    )


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a new PDF document"""
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Save file
    file_path = PAPERS_DIR / file.filename
    content = await file.read()
    
    with open(file_path, "wb") as f:
        f.write(content)
    
    return {
        "filename": file.filename,
        "size": len(content),
        "message": "File uploaded successfully"
    }


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a PDF document"""
    file_path = PAPERS_DIR / filename
    
    if not file_path.exists() or not str(file_path).startswith(str(PAPERS_DIR)):
        raise HTTPException(status_code=404, detail="Document not found")
    
    file_path.unlink()
    return {"message": f"Document {filename} deleted successfully"}


@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat with streaming responses
    
    Message format:
    - Client sends: {"query": "user question"}
    - Server streams: {"type": "node_enter", "node": "node_name"}
    -                 {"type": "content", "data": "streamed text"}
    -                 {"type": "node_exit", "node": "node_name"}
    -                 {"type": "done"}
    """
    await websocket.accept()
    thread_id = new_uuid()
    thread = {"configurable": {"thread_id": thread_id}}
    prev_node = None
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            query = message.get("query", "").strip()
            
            if not query:
                await websocket.send_json({
                    "type": "error",
                    "message": "Query cannot be empty"
                })
                continue
            
            # Send query received confirmation
            await websocket.send_json({
                "type": "query_received",
                "thread_id": thread_id
            })
            
            # Process query with streaming
            try:
                input_state = InputState(messages=query, user_question=query)
                prev_node = None
                
                async for c, metadata in graph.astream(
                    input=input_state,
                    stream_mode="messages",
                    config=thread
                ):
                    # Handle node transitions
                    node = metadata.get("langgraph_node") or metadata.get("step")
                    if node != prev_node:
                        if prev_node is not None:
                            await websocket.send_json({
                                "type": "node_exit",
                                "node": prev_node
                            })
                        if node is not None:
                            await websocket.send_json({
                                "type": "node_enter",
                                "node": node
                            })
                        prev_node = node
                    
                    # Stream content
                    if c.content:
                        await websocket.send_json({
                            "type": "content",
                            "data": c.content
                        })
                
                # Final node exit
                if prev_node is not None:
                    await websocket.send_json({
                        "type": "node_exit",
                        "node": prev_node
                    })
                
                # Send completion signal
                await websocket.send_json({
                    "type": "done"
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


@app.post("/chat")
async def post_chat(request: QueryRequest):
    """
    HTTP endpoint for chat (non-streaming)
    For streaming, use WebSocket endpoint instead
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    thread_id = request.thread_id or new_uuid()
    thread = {"configurable": {"thread_id": thread_id}}
    
    try:
        input_state = InputState(messages=query, user_question=query)
        response_content = ""
        
        async for c, metadata in graph.astream(
            input=input_state,
            stream_mode="messages",
            config=thread
        ):
            if c.content:
                response_content += c.content
        
        return {
            "response": response_content,
            "thread_id": thread_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
