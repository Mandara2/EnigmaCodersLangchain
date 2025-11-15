"""
API REST para RAG Jurídico - Constitución Política de Colombia
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uvicorn

from config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    CORS_ORIGINS
)
from rag_engine import get_rag_engine


# ====================================
# MODELOS PYDANTIC
# ====================================

class QueryRequest(BaseModel):
    """Modelo para solicitud de consulta"""
    question: str = Field(
        ..., 
        min_length=3,
        max_length=500,
        description="Pregunta sobre la Constitución"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "¿Qué es la acción de tutela?"
            }
        }


class Source(BaseModel):
    """Modelo para fuente citada"""
    page: Any = Field(..., description="Número de página")
    content: str = Field(..., description="Fragmento del contenido")


class QueryResponse(BaseModel):
    """Modelo para respuesta de consulta"""
    answer: str = Field(..., description="Respuesta generada")
    sources: List[Source] = Field(..., description="Fuentes citadas")
    question: str = Field(..., description="Pregunta original")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "La acción de tutela es un mecanismo constitucional...",
                "sources": [
                    {
                        "page": 42,
                        "content": "Artículo 86. Toda persona tendrá acción de tutela..."
                    }
                ],
                "question": "¿Qué es la acción de tutela?"
            }
        }


class HealthResponse(BaseModel):
    """Modelo para health check"""
    status: str
    message: str
    vectorstore_ready: bool


# ====================================
# FASTAPI APP
# ====================================

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====================================
# INICIALIZACIÓN
# ====================================

@app.on_event("startup")
async def startup_event():
    """Inicializa el RAG Engine al arrancar"""
    print("\n" + "=" * 70)
    print("🚀 INICIANDO API RAG JURÍDICO")
    print("=" * 70)
    try:
        get_rag_engine()
        print("✅ API lista para recibir consultas")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"❌ Error al inicializar: {e}")
        print("=" * 70 + "\n")


# ====================================
# ENDPOINTS
# ====================================

@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz"""
    return {
        "message": "RAG Jurídico - Constitución de Colombia API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Verifica el estado de la API
    """
    try:
        engine = get_rag_engine()
        vectorstore_ready = engine.vectorstore is not None
        
        return HealthResponse(
            status="healthy",
            message="API funcionando correctamente",
            vectorstore_ready=vectorstore_ready
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            message=f"Error: {str(e)}",
            vectorstore_ready=False
        )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_constitution(request: QueryRequest):
    """
    Realiza una consulta sobre la Constitución Política de Colombia
    
    - **question**: Pregunta sobre cualquier tema constitucional
    
    Retorna la respuesta generada con las fuentes citadas.
    """
    try:
        # Obtener RAG engine
        engine = get_rag_engine()
        
        # Realizar consulta
        result = engine.query(request.question)
        
        # Convertir a modelo de respuesta
        return QueryResponse(
            answer=result["answer"],
            sources=[Source(**source) for source in result["sources"]],
            question=result["question"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar consulta: {str(e)}"
        )


@app.post("/rebuild-vectorstore", tags=["Admin"])
async def rebuild_vectorstore():
    """
    Reconstruye el vectorstore desde cero
    
    ⚠️ Esta operación puede tardar varios minutos
    """
    try:
        engine = get_rag_engine()
        engine.rebuild_vectorstore()
        return {
            "status": "success",
            "message": "Vectorstore reconstruido exitosamente"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al reconstruir vectorstore: {str(e)}"
        )


# ====================================
# EJEMPLOS DE CONSULTA
# ====================================

@app.get("/examples", tags=["General"])
async def get_examples():
    """
    Obtiene ejemplos de consultas
    """
    return {
        "examples": [
            "¿Qué es la acción de tutela?",
            "¿Cuáles son los derechos fundamentales?",
            "¿Qué dice el artículo 1 de la constitución?",
            "¿Cómo se reforma la constitución?",
            "¿Qué es el habeas corpus?",
            "¿Cuáles son las ramas del poder público?",
            "¿Qué dice la constitución sobre la educación?",
            "¿Cómo funciona el sistema judicial colombiano?",
        ]
    }


# ====================================
# RUN SERVER
# ====================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🏛️  RAG JURÍDICO - CONSTITUCIÓN DE COLOMBIA")
    print("=" * 70)
    print("📚 Documentación: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("💬 Query Endpoint: POST http://localhost:8000/query")
    print("=" * 70 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )