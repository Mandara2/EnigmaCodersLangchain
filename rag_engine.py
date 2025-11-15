"""
Motor RAG para consultas sobre la Constitución Política de Colombia
"""
import os
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVER_K,
    PDF_PATH,
    VECTORSTORE_DIR,
    SYSTEM_PROMPT
)


class RAGEngine:
    """Motor RAG para consultas sobre documentos jurídicos"""
    
    def __init__(self):
        """Inicializa el motor RAG"""
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.rag_chain = None
        
        print("🚀 Inicializando RAG Engine...")
        self._initialize()
        print("✅ RAG Engine listo!")
    
    def _initialize(self):
        """Inicializa todos los componentes del RAG"""
        # 1. Configurar LLM
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE
        )
        
        # 2. Configurar embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL
        )
        
        # 3. Cargar o crear vectorstore
        if self._vectorstore_exists():
            print("📦 Cargando vectorstore existente...")
            self._load_vectorstore()
        else:
            print("📄 Vectorstore no encontrado. Procesando PDF...")
            self._create_vectorstore()
        
        # 4. Crear cadena RAG
        self._create_rag_chain()
    
    def _vectorstore_exists(self) -> bool:
        """Verifica si el vectorstore ya existe"""
        return (VECTORSTORE_DIR / "chroma.sqlite3").exists()
    
    def _load_vectorstore(self):
        """Carga el vectorstore existente"""
        self.vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=self.embeddings
        )
    
    def _create_vectorstore(self):
        """Crea el vectorstore desde el PDF"""
        # Verificar que existe el PDF
        if not PDF_PATH.exists():
            raise FileNotFoundError(
                f"❌ PDF no encontrado en: {PDF_PATH}\n"
                f"Por favor, coloca 'constitucion_colombia.pdf' en la carpeta 'data/'"
            )
        
        # 1. Cargar PDF
        print(f"📖 Cargando PDF: {PDF_PATH.name}")
        loader = PyPDFLoader(str(PDF_PATH))
        documents = loader.load()
        print(f"   ✓ {len(documents)} páginas cargadas")
        
        # 2. Dividir en fragmentos
        print("✂️  Dividiendo en fragmentos...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(documents)
        print(f"   ✓ {len(splits)} fragmentos creados")
        
        # 3. Crear vectorstore
        print("🔢 Creando embeddings y vectorstore...")
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=str(VECTORSTORE_DIR)
        )
        print("   ✓ Vectorstore creado y guardado")
    
    def _create_rag_chain(self):
        """Crea la cadena RAG"""
        # 1. Crear retriever
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": RETRIEVER_K}
        )
        
        # 2. Crear prompt
        prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        
        # 3. Crear cadenas
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(retriever, document_chain)
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Realiza una consulta al RAG
        
        Args:
            question: Pregunta del usuario
            
        Returns:
            Dict con 'answer' y 'sources'
        """
        print(f"\n💬 Consulta: {question}")
        
        # Ejecutar RAG
        response = self.rag_chain.invoke({"input": question})
        
        # Extraer fuentes
        sources = []
        for doc in response.get("context", []):
            sources.append({
                "page": doc.metadata.get("page", "N/A"),
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        
        result = {
            "answer": response["answer"],
            "sources": sources,
            "question": question
        }
        
        print(f"✅ Respuesta generada ({len(sources)} fuentes)")
        return result
    
    def rebuild_vectorstore(self):
        """Reconstruye el vectorstore desde cero"""
        print("🔄 Reconstruyendo vectorstore...")
        
        # Eliminar vectorstore existente
        import shutil
        if VECTORSTORE_DIR.exists():
            shutil.rmtree(VECTORSTORE_DIR)
            VECTORSTORE_DIR.mkdir()
        
        # Recrear
        self._create_vectorstore()
        self._create_rag_chain()
        
        print("✅ Vectorstore reconstruido")


# ====================================
# INSTANCIA GLOBAL (Singleton)
# ====================================
_rag_instance = None

def get_rag_engine() -> RAGEngine:
    """Obtiene la instancia única del RAG Engine"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGEngine()
    return _rag_instance


# ====================================
# TESTING
# ====================================
if __name__ == "__main__":
    # Prueba del motor RAG
    print("=" * 70)
    print("🧪 PRUEBA DEL MOTOR RAG")
    print("=" * 70)
    
    try:
        engine = get_rag_engine()
        
        # Consulta de prueba
        result = engine.query("¿Qué es la acción de tutela?")
        
        print("\n" + "=" * 70)
        print("📝 RESPUESTA:")
        print("=" * 70)
        print(result["answer"])
        
        print("\n" + "=" * 70)
        print("📚 FUENTES:")
        print("=" * 70)
        for i, source in enumerate(result["sources"], 1):
            print(f"\n--- Fuente {i} (Página {source['page']}) ---")
            print(source["content"])
        
        print("\n" + "=" * 70)
        print("✅ PRUEBA COMPLETADA")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")