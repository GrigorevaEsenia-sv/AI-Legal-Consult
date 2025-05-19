from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_compressa import ChatCompressa


class TSACashier:
    documents = [
        # "Что-нибудь еще?",
        # "Добрый день! Вам с собой или здесь?",
        # "К оплате _ рублей. Оплата картой или наличными?",
        # "Приятного аппетита! Заходите снова."
        # "Что-нибудь еще?",
        # "Ваш заказ: _. Всё верно?",
        # " _ добавлен в заказ. Что-нибудь еще?",
        # "Что-нибудь еще?",
        "Something else?"
    ]

    def __init__(self, api_key, role):
        self.api_key = api_key
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        docs = text_splitter.split_text(" ".join(self.documents))

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_texts(docs, embeddings)

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