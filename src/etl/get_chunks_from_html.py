from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_and_chunk_html_documents(file_path, chunk_size=500, chunk_overlap=50):
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
    with open(file_path, 'r', encoding="utf-8") as file:
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
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=['\n\n', '\n', ';', '. ', ' ', '']
    )
    
    # Нарезка на чанки
    chunks = []
    metadatas = []
    
    for section in sections:
        section_chunks = text_splitter.split_text(section)
        
        # Извлечение метаданных (раздел, глава, статья) из текста
        for chunk in section_chunks:
            metadata = {}
            lines = chunk.split('\n')
            
            for line in lines:
                if line.startswith('РАЗДЕЛ:'):
                    metadata['section'] = line.replace('РАЗДЕЛ:', '').strip()
                elif line.startswith('ГЛАВА:'):
                    metadata['chapter'] = line.replace('ГЛАВА:', '').strip()
                elif line.startswith('СТАТЬЯ'):
                    metadata['article'] = line.strip()
            
            chunks.append(chunk)
            metadatas.append(metadata)
    
    for chunk in chunks[5]:
        print(chunk[:300])
    # Инициализация эмбеддингов
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    

    # Создание векторного хранилища
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas
    )
    
    return vectorstore

# Пример использования
if __name__ == "__main__":
    doc_path = 'data/raw/housing_code/JKRF.html'
    print('h'*50)
    # Загрузка документа и создание векторного хранилища
    vectorstore = load_and_chunk_html_documents(doc_path)
    