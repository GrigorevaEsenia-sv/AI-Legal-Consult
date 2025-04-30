from langchain.document_loaders import TextLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_compressa import ChatCompressa
# from data_loader import load_data
from docx import Document
import re


from docx import Document
from typing import List, Tuple
import re

# Helper function to extract the last heading from existing chunks
def get_last_heading(chunks: List[str]) -> str:
    """Extracts the most recent heading from generated chunks"""
    if not chunks:
        return ""
    
    # Search backwards through chunks to find last heading
    for chunk in reversed(chunks):
        heading_match = re.search(r'### (.+)', chunk)
        if heading_match:
            return heading_match.group(1)
    return ""

def extract_structural_elements(doc_path: str) -> List[Tuple[str, str]]:
    """Extracts headings and text paragraphs from Word document"""
    doc = Document(doc_path)
    elements = []
    for para in doc.paragraphs:
        if para.style.name.startswith('Heading'):
            elements.append(("heading", para.text))
        elif para.text.strip():
            elements.append(("text", para.text))
    return elements

def hierarchical_chunking(elements: List[Tuple[str, str]], max_chunk: int = 500) -> List[str]:
    """Chunks document while preserving legal article structure"""
    chunks = []
    current_chunk = []
    current_size = 0

    for elem_type, text in elements:
        elem_size = len(text.split())
        
        if elem_type == "heading":
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
            current_chunk.append(f"### {text}")
            current_size = elem_size
        else:
            if current_size + elem_size > max_chunk and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [f"### {get_last_heading(chunks)}"] if get_last_heading(chunks) else []
                current_size = len(" ".join(current_chunk).split())
            current_chunk.append(text)
            current_size += elem_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


class LegalConsult:
    doc_path = 'data/raw/housing_code/JKRF.docx'

    # Создание embeddings и хранение в Compressa Vector DB
    def __init__(self, api_key, role):
        self.api_key = api_key
        elements = extract_structural_elements(self.doc_path)
        chunks = hierarchical_chunking(elements)

        #Проверка
        print('??????????????????????????????????????????????')
        print(chunks[0])

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_texts(chunks, embeddings)

        self.llm = ChatCompressa(
            base_url="https://compressa-api.mil-team.ru/v1",
            api_key=api_key,
            temperature=0.2,
            max_tokens=50,
            stream="false"
        )

        self.messages = [
            ("system",
             role),
        ]

    def get_answer(self, client_answer):
        # Извлечение релевантной информации из базы знаний
        docs = self.vectorstore.similarity_search(client_answer, k=2)  # Ищем 2 наиболее релевантных фрагмента
        context = " ".join([doc.page_content for doc in docs])

        # Добавляем контекст в сообщения
        self.messages.append(("system", f"Context: {context}"))
        self.messages.append(("human", client_answer))

        # Генерация ответа
        ai_msg = self.llm.invoke(self.messages)

        # Добавляем ответ в историю
        self.messages.append(("assistant", ai_msg.content))

        return ai_msg.content
