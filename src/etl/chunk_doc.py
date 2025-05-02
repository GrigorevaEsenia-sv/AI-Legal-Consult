import re
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class LegalChunk:
    text: str
    metadata: Dict[str, str]  # Название закона, статья, пункт и т.д.

def chunk_legal_document(
    text: str,
    doc_name: str = "",
    min_chunk_size: int = 500,
    max_chunk_size: int = 1000,
    overlap: int = 50
) -> List[LegalChunk]:
    """
    Нарезает юридический документ на смысловые чанки с сохранением структуры.
    
    Параметры:
        text: Текст документа для обработки
        doc_name: Название закона/документа
        min_chunk_size: Минимальный размер чанка (символов)
        max_chunk_size: Максимальный размер чанка
        overlap: Перекрытие между чанками (символов)
    
    Возвращает:
        Список объектов LegalChunk с текстом и метаданными
    """
    
    # 1. Предварительная очистка текста
    text = clean_legal_text(text)
    
    # 2. Разбиение на структурные элементы (статьи, пункты)
    structural_units = split_into_structural_units(text, doc_name)
    
    chunks = []
    
    for unit in structural_units:
        unit_text = unit['text']
        unit_meta = unit['metadata']
        
        # 3. Дополнительное разбиение больших статей на абзацы/подпункты
        if len(unit_text) > max_chunk_size:
            sub_units = split_large_unit(unit_text, min_chunk_size, max_chunk_size, overlap)
            
            for i, sub_text in enumerate(sub_units):
                # Сохраняем связь с родительской структурой
                sub_meta = unit_meta.copy()
                sub_meta['part'] = i + 1
                
                chunks.append(LegalChunk(
                    text=format_chunk_text(sub_text, sub_meta),
                    metadata=sub_meta
                ))
        else:
            chunks.append(LegalChunk(
                text=format_chunk_text(unit_text, unit_meta),
                metadata=unit_meta
            ))
    
    return chunks

def clean_legal_text(text: str) -> str:
    """Очистка юридического текста от артефактов"""
    # Удаление номеров страниц (например: "\n[12]\n")
    text = re.sub(r'\n\[\d+\]\n', '\n', text)
    # Удаление HTML-тегов если есть
    text = re.sub(r'<[^>]+>', '', text)
    # Нормализация пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_structural_units(text: str, doc_name: str) -> List[Dict]:
    """Разбивает текст на структурные единицы (статьи, главы)"""
    units = []
    
    # Регулярка для статей с пунктами (пример для российских законов)
    article_pattern = re.compile(
        r'(Статья\s+\d+\.\s*(.*?)\n)(.*?)(?=(Статья\s+\d+\.|$)',
        re.DOTALL
    )
    
    for match in article_pattern.finditer(text):
        title = match.group(1).strip()
        content = match.group(3).strip()
        
        # Извлекаем номер статьи
        article_num = re.search(r'Статья\s+(\d+)', title).group(1)
        
        units.append({
            'text': f"{title}\n{content}",
            'metadata': {
                'law': doc_name,
                'article': article_num,
                'title': match.group(2).strip() if match.group(2) else ""
            }
        })
    
    # Если не найдены статьи, разбиваем по абзацам
    if not units:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        for i, para in enumerate(paragraphs):
            units.append({
                'text': para,
                'metadata': {
                    'law': doc_name,
                    'part': i + 1
                }
            })
    
    return units

def split_large_unit(text: str, min_size: int, max_size: int, overlap: int) -> List[str]:
    """Разбивает большой структурный элемент на чанки"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 для пробела
        
        if current_length >= min_size and word.endswith(('.', ';')):
            chunks.append(' '.join(current_chunk))
            # Сохраняем перекрытие
            current_chunk = current_chunk[-overlap:] if overlap else []
            current_length = sum(len(w) + 1 for w in current_chunk)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Объединяем слишком короткие чанки
    merged_chunks = []
    buffer = []
    buffer_length = 0
    
    for chunk in chunks:
        chunk_length = len(chunk)
        if buffer_length + chunk_length <= max_size:
            buffer.append(chunk)
            buffer_length += chunk_length
        else:
            if buffer:
                merged_chunks.append(' '.join(buffer))
            buffer = [chunk]
            buffer_length = chunk_length
    
    if buffer:
        merged_chunks.append(' '.join(buffer))
    
    return merged_chunks

def format_chunk_text(text: str, metadata: Dict) -> str:
    """Форматирует текст чанка с включением метаданных"""
    law = metadata.get('law', '')
    article = metadata.get('article', '')
    title = metadata.get('title', '')
    
    header = []
    if law:
        header.append(law)
    if article:
        header.append(f"Статья {article}")
    if title:
        header.append(title)
    
    if header:
        return f"{': '.join(header)}\n\n{text}"
    return text