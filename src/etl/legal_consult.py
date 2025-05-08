from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_compressa import ChatCompressa
# from data_loader import load_data
from tqdm import tqdm
from get_chunks_from_html import load_and_chunk_html_documents
from typing import List
import re


class LegalConsult:
    doc_path = 'data/raw/housing_code/garant/1.html'

    def __init__(self, api_key, role):
        self.api_key = api_key

        chunks, metadatas = tqdm(load_and_chunk_html_documents(self.doc_path))

        # Инициализация эмбеддингов
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Создание векторного хранилища
        self.vectorstore = FAISS.from_texts(
            texts=chunks,
            embedding=embeddings,
            metadatas=metadatas
        )
    
        # Сохранение векторного хранилища для последующего использования
        self.vectorstore.save_local("legal_docs_faiss_index")

        # Загрузка при необходимости
        # vectorstore = FAISS.load_local("legal_docs_faiss_index", embeddings)

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


# Пример использования
if __name__ == "__main__":
    doc_path = 'data/raw/housing_code/garant/1.html'
    print('h'*50)
    # Загрузка документа и создание векторного хранилища
    chunks, metadatas = tqdm(load_and_chunk_html_documents(doc_path))

    for i in range(4):
        print(metadatas[i])
        print(chunks[i])
        print('='*50)


    # Инициализация эмбеддингов
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={'normalize_embeddings': True}
    )
    
    
    # Создание векторного хранилища
    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas
    )

    # Пример поиска
    query = "Перепланировка квартиры. Я хочу установить душевую кабинку вместо ванной, являюсь собственником квартиры. Могу ли я это сделать без каких-либо разрешений?"
    docs = tqdm(vectorstore.similarity_search(query, k=2), desc='similarity_search in vectorestore')
    
    print(f"query: {query}")
    for doc in docs:
        print("Текст:", doc.page_content)
    