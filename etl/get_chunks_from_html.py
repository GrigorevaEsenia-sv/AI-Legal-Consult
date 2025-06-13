from bs4 import BeautifulSoup
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
from datetime import datetime
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import html

def clean_text(text):
    """
    Очистка текста от HTML-сущностей и специальных пробелов с сохранением переносов строк.
    """
    if not text:
        return text
    
    # Декодируем HTML-сущности
    text = html.unescape(text)
    
    # Заменяем специальные пробелы на обычные, сохраняя переносы строк
    text = re.sub(r'[\xa0\u200b\u202f]+', ' ', text)
    
    return text.lstrip('.;, ')

def load_and_chunk_html_documents(file_path, chunk_size=1600, chunk_overlap=150):
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
    for tag in soup(['style', 'script', 'meta', 'link', 'noscript', 'iframe', 'svg']):
        tag.decompose()
    
    # Извлечение структурированного текста
    sections = []
    current_section = ""
    
    # Обработка структуры документа (разделы, главы, статьи)
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'div', 'article', 'section']):
        element_text = clean_text(element.get_text())
        if not element_text:
            continue
            
        if element.name in ['h1', 'h2', 'h3', 'h4']:
            if current_section:
                sections.append(current_section.strip())
                current_section = ""
            current_section += f"\n{element_text.upper()}\n"
        else:
            current_section += element_text + " "
    
    if current_section:
        sections.append(current_section.strip())
    
    # Настройка сплиттера для юридических текстов
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=['\n\n', '\n', ';', '. ']
    )
    
    # Нарезка на чанки
    chunks = []
    metadatas = []

    metadata = dict()

    print(f'len(sections) {len(sections)}')
    for section in sections:
        section_chunks = text_splitter.split_text(section)
        
        # Извлечение метаданных (раздел, глава, статья) из текста
        print(f'len(section_chunks) {len(section_chunks)}')
        for chunk in section_chunks:
            current_metadata = metadata.copy()
            lines = chunk.split('\n')
            for line in lines:
                line = clean_text(line)

                if line.startswith('Раздел'):
                    current_metadata['section'] = line.strip()
                elif line.startswith('Глава'):
                    current_metadata['chapter'] = line.strip()
                elif line.startswith('Статья'):
                    current_metadata['article'] = line.strip()

            chunks.append(chunk)
            metadatas.append(metadata)
            metadata = current_metadata

    return chunks, metadatas

if __name__ == "__main__":
    doc_path = 'data/raw/housing_code/garant/1.html'

    # Создаем уникальное имя файла с timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    result_dir = 'data/processed/chunks_examples_from_garant'
    result_filename = f"chunks_{timestamp}_{unique_id}.txt"
    result_path = os.path.join(result_dir, result_filename)

    # Загрузка документа и создание векторного хранилища
    chunks, metadatas = tqdm(load_and_chunk_html_documents(doc_path))


    # # Создаем директорию если нужно
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    with open(result_path, 'w', encoding='utf-8') as f:  
        for i in range(len(chunks)):
            f.write('\nmetadatas\n')
            f.write(str(metadatas[i]))
            f.write('\n')
            f.write(chunks[i])

    