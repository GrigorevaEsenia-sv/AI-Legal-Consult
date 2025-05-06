from bs4 import BeautifulSoup
import re
from langchain.vectorstores import FAISS
from typing import List, Dict, Tuple, Optional
import os
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

def load_and_chunk_html_documents(file_path: str) -> Tuple[List[str], List[Dict]]:
    """
    Load an HTML legal document and chunk it according to the legal structure.
    Each chunk starts with a section, chapter, article, and then contains one point.
    
    Args:
        file_path: Path to the HTML file containing legal document
        
    Returns:
        Tuple containing:
        - List of text chunks
        - List of metadata dictionaries for each chunk
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load the HTML document
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract document structure
    sections = extract_sections(soup)
    
    # Initialize result containers
    texts = []
    metadatas = []
    
    # Process each section
    for section in sections:
        section_title = section.get('title', 'НЕИЗВЕСТНЫЙ РАЗДЕЛ')
        
        # Process chapters in this section
        chapters = extract_chapters(section)
        for chapter in chapters:
            chapter_title = chapter.get('title', 'НЕИЗВЕСТНАЯ ГЛАВА')
            
            # Process articles in this chapter
            articles = extract_articles(chapter)
            for article in articles:
                article_title = article.get('title', 'НЕИЗВЕСТНАЯ СТАТЬЯ')
                article_number = extract_article_number(article_title)
                
                # Extract the main text of the article
                article_text = extract_article_text(article)
                
                # Extract points from the article
                points = extract_points(article)
                
                if points:
                    # Create a chunk for each point
                    for point_num, point_text in points:
                        chunk_text = format_chunk(
                            section_title, 
                            chapter_title, 
                            article_title, 
                            article_text, 
                            point_num, 
                            point_text
                        )
                        
                        metadata = {
                            'section': section_title,
                            'chapter': chapter_title,
                            'article': article_title,
                            'article_number': article_number,
                            'point_number': point_num,
                            'source': file_path
                        }
                        
                        texts.append(chunk_text)
                        metadatas.append(metadata)
                else:
                    # If no points, create a single chunk for the article
                    chunk_text = format_chunk(
                        section_title, 
                        chapter_title, 
                        article_title, 
                        article_text, 
                        None, 
                        None
                    )
                    
                    metadata = {
                        'section': section_title,
                        'chapter': chapter_title,
                        'article': article_title,
                        'article_number': article_number,
                        'point_number': None,
                        'source': file_path
                    }
                    
                    texts.append(chunk_text)
                    metadatas.append(metadata)
    
    return texts, metadatas

def extract_sections(soup: BeautifulSoup) -> List[Dict]:
    """
    Extract sections from the document.
    This function needs to be adapted based on the actual HTML structure.
    """
    # Find section elements based on your HTML structure
    # This is a placeholder - you'll need to modify based on actual HTML
    section_elements = soup.find_all('div', class_='section')
    
    sections = []
    for element in section_elements:
        title_element = element.find('h1') or element.find('h2')
        title = title_element.text.strip() if title_element else 'НЕИЗВЕСТНЫЙ РАЗДЕЛ'
        sections.append({
            'title': title,
            'element': element
        })
    
    return sections

def extract_chapters(section: Dict) -> List[Dict]:
    """
    Extract chapters from a section.
    This function needs to be adapted based on the actual HTML structure.
    """
    # Find chapter elements based on your HTML structure
    # This is a placeholder - you'll need to modify based on actual HTML
    chapter_elements = section['element'].find_all('div', class_='chapter')
    
    chapters = []
    for element in chapter_elements:
        title_element = element.find('h2') or element.find('h3')
        title = title_element.text.strip() if title_element else 'НЕИЗВЕСТНАЯ ГЛАВА'
        chapters.append({
            'title': title,
            'element': element
        })
    
    return chapters

def extract_articles(chapter: Dict) -> List[Dict]:
    """
    Extract articles from a chapter.
    This function needs to be adapted based on the actual HTML structure.
    """
    # Find article elements based on your HTML structure
    # This is a placeholder - you'll need to modify based on actual HTML
    article_elements = chapter['element'].find_all('div', class_='article')
    
    articles = []
    for element in article_elements:
        title_element = element.find('h3') or element.find('h4')
        title = title_element.text.strip() if title_element else 'НЕИЗВЕСТНАЯ СТАТЬЯ'
        articles.append({
            'title': title,
            'element': element
        })
    
    return articles

def extract_article_number(article_title: str) -> Optional[str]:
    """Extract article number from the article title."""
    match = re.search(r'Статья\s+(\d+)', article_title)
    return match.group(1) if match else None

def extract_article_text(article: Dict) -> str:
    """
    Extract the main text of an article, excluding points.
    This function needs to be adapted based on the actual HTML structure.
    """
    # This is a placeholder - you'll need to modify based on actual HTML
    text_element = article['element'].find('div', class_='article-text')
    if text_element:
        # Remove point elements if they're nested inside the text element
        for point_element in text_element.find_all('div', class_='point'):
            point_element.decompose()
        return text_element.text.strip()
    return ""

def extract_points(article: Dict) -> List[Tuple[str, str]]:
    """
    Extract numbered points from an article.
    This function needs to be adapted based on the actual HTML structure.
    
    Returns:
        List of tuples (point_number, point_text)
    """
    # Find point elements based on your HTML structure
    # This is a placeholder - you'll need to modify based on actual HTML
    point_elements = article['element'].find_all('div', class_='point')
    
    points = []
    for element in point_elements:
        # Extract point number - adapt pattern based on your document structure
        number_match = re.search(r'(\d+)[\.|\)]', element.text)
        number = number_match.group(1) if number_match else 'N/A'
        
        # Extract point text
        text = element.text.strip()
        
        points.append((number, text))
    
    # Alternative pattern matching approach for when points aren't in separate elements
    if not points:
        article_text = article['element'].text
        # Match patterns like "1. Text" or "1) Text"
        point_matches = re.finditer(r'(\d+)[\.\)]\s+([^\d].*?)(?=\d+[\.\)]|$)', article_text, re.DOTALL)
        for match in point_matches:
            number = match.group(1)
            text = match.group(2).strip()
            points.append((number, text))
    
    return points

def format_chunk(section: str, chapter: str, article: str, article_text: str, 
                point_num: Optional[str], point_text: Optional[str]) -> str:
    """
    Format a chunk according to the specified structure.
    
    Args:
        section: Section title
        chapter: Chapter title
        article: Article title
        article_text: Main text of the article
        point_num: Point number (if any)
        point_text: Point text (if any)
        
    Returns:
        Formatted chunk text
    """
    chunk = f"{{Раздел: {section}\n"
    chunk += f"Глава: {chapter}\n"
    chunk += f"Статья: {article}\n"
    chunk += f"Текст:\n{article_text}\n"
    
    if point_num and point_text:
        chunk += f"{point_num}. {point_text}}}"
    else:
        chunk += "}"
    
    return chunk

def create_vectorstore_from_html_documents(doc_path: str, embeddings) -> FAISS:
    """
    Create a FAISS vectorstore from HTML legal documents.
    
    Args:
        doc_path: Path to the HTML document
        embeddings: Embedding model to use
        
    Returns:
        FAISS vectorstore
    """
    texts, metadatas = load_and_chunk_html_documents(doc_path)
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    return vectorstore

# Example usage:
# from langchain.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectorstore = create_vectorstore_from_html_documents("housing_code.html", embeddings)
# Пример использования
if __name__ == "__main__":
    doc_path = 'data/raw/housing_code/JKRF.html'
    print('h'*50)
    # Загрузка документа и создание векторного хранилища
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = load_and_chunk_html_documents(doc_path, embeddings)
    
    # Пример поиска
    query = "условия осуществления права на жилище"
    docs = tqdm(vectorstore.similarity_search(query, k=2), desc='similarity_search in vectorestore')
    
    for doc in docs:
        print("Текст:", doc.page_content[:200] + "...")