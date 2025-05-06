from bs4 import BeautifulSoup
import re
from typing import List, Dict, Tuple, Optional
import os
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
from langchain.schema import Document

from bs4 import BeautifulSoup
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_html_documents(file_path, chunk_overlap=50):
    """
    Загружает HTML документ, извлекает текст, нарезает на чанки и создает векторное хранилище.
    
    Args:
        file_path (str): Путь к HTML файлу
        chunk_size (int): Размер чанка в символах
        chunk_overlap (int): Перекрытие между чанками
        
    Returns:
        FAISS: Векторное хранилище с чанками документа
    """
    # Загрузка и парсинг HTML
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Удаление ненужных тегов (стили, скрипты)
    for tag in soup(['style', 'script', 'meta', 'link']):
        tag.decompose()
    
    # Извлечение структурированного текста
    sections = []
    current_section = ""
    
    # Обработка структуры документа (разделы, главы, статьи)
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'div']):
        if element.name in ['h1', 'h2', 'h3', 'h4']:
            if current_section:
                sections.append(current_section.strip())
                current_section = ""
            current_section += f"\n{element.get_text().upper()}\n"
        else:
            current_section += element.get_text() + " "
    
    if current_section:
        sections.append(current_section.strip())
    
    # Настройка сплиттера для юридических текстов
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=['\n\n', '\n', ';', ' ', '']
    )
    
    # Нарезка на чанки
    chunks = []
    metadatas = []
    
    for section in sections:
        section_chunks = text_splitter.split_text(section)
        
        # Извлечение метаданных (раздел, глава, статья) из текста
        for chunk in section_chunks:
            chunk = chunk.lstrip('.;, ')
            # print('='*50)
            # print(chunk)
            # print('='*50)

            metadata = {}
            lines = chunk.split('\n')
            
            for line in lines:
                if line.startswith('Раздел'):
                    metadata['section'] = line.replace('Раздел', '').strip()
                elif line.startswith('Глава'):
                    metadata['chapter'] = line.replace('Глава', '').strip()
                elif line.startswith('Статья'):
                    metadata['article'] = line.strip()
            
            chunks.append(chunk)
            metadatas.append(metadata)
            print('='*50)
            print(metadata)
            print(chunk)
            print('='*50)
    
    # Инициализация эмбеддингов
    embeddings = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2")
    
    # Создание векторного хранилища
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas
    )
    
    return vectorstore
if __name__ == "__main__":
    doc_path = 'data/raw/housing_code/garant/1.html'
    print('h'*50)
    # Загрузка документа и создание векторного хранилища
    texts, metadatas = load_and_chunk_html_documents(doc_path)
    # print(texts)
    # for text in texts[:20]:
    #     print(text)
    #     print('='*50)
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # vectorstore = FAISS.from_texts(
    #     texts=texts, 
    #     embedding=embeddings, 
    #     metadatas=metadatas
    # )
    
    # # Пример поиска
    # query = "условия осуществления права на жилище"
    # docs = tqdm(vectorstore.similarity_search(query, k=2), desc='similarity_search in vectorestore')
    
    # for doc in docs:
    #     print("Текст:", doc.page_content[:200] + "...")