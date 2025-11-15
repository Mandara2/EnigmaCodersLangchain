"""
Configuración de la aplicación RAG Jurídico
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# ====================================
# PATHS
# ====================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Crear directorios si no existen
DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)

# ====================================
# API KEYS
# ====================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "❌ GOOGLE_API_KEY no encontrada. "
        "Crea un archivo .env con: GOOGLE_API_KEY=tu_key"
    )

# ====================================
# MODELO LLM
# ====================================
LLM_MODEL = "gemini-1.5-flash"
LLM_TEMPERATURE = 0

# ====================================
# EMBEDDINGS
# ====================================
EMBEDDING_MODEL = "models/text-embedding-004"  # Ya lo tienes

# ====================================
# RAG CONFIG
# ====================================
CHUNK_SIZE = 5000      # Chunks EXTRA grandes
CHUNK_OVERLAP = 1000   # Overlap grande
RETRIEVER_K = 3  # Número de fragmentos a recuperar

# ====================================
# PDF CONFIG
# ====================================
PDF_PATH = DATA_DIR / "constitucion_colombia.pdf"

# ====================================
# PROMPTS
# ====================================
SYSTEM_PROMPT = """Eres un abogado constitucionalista experto en la Constitución Política de Colombia.

Tu función es responder preguntas sobre la constitución basándote ÚNICAMENTE en los fragmentos proporcionados.

Reglas importantes:
1. Cita siempre el número de artículo cuando sea relevante
2. Si la información no está en el contexto, di claramente que no la tienes
3. Sé preciso y usa lenguaje jurídico apropiado pero comprensible
4. Cuando cites un artículo, menciona su número completo (ej: "Artículo 86")

Contexto de los fragmentos relevantes:
{context}

Pregunta del usuario:
{question}

Respuesta (incluye citas a artículos específicos):"""

# ====================================
# API CONFIG
# ====================================
API_TITLE = "RAG Jurídico - Constitución de Colombia"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
API para consultas sobre la Constitución Política de Colombia usando RAG.

## Características:
- 🔍 Búsqueda semántica en la constitución
- 📜 Citación automática de artículos
- 🤖 Respuestas generadas con IA (Gemini)
"""

# ====================================
# CORS
# ====================================
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

# ====================================
# LOGGING
# ====================================
LOG_LEVEL = "INFO"