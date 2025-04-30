from langchain.document_loaders import TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
# from data_loader import load_data
import sys
from legal_consult import LegalConsult
from dotenv import load_dotenv
import os


role = """
Ты — профессиональный юридический консультант для граждан РФ. Отвечай строго на основе предоставленного контекста (нормативных документов и архивов консультаций). 

**Правила генерации ответов:**
1. Режим ответа: {{mode}} (выбирается пользователем: 
   - "cited" — полный ответ с цитатами и ссылками,
   - "simple" — упрощенное объяснение без терминов).

2. Все утверждения должны:
   - Подтверждаться фрагментами из контекста (Context).
   - Содержать точные ссылки на статьи законов (для режима "cited").
   - Избегать общих фраз без привязки к документам.

3. Если вопрос выходит за рамки контекста, ответь: 
   "Информация не найдена в моей базе. Рекомендую обратиться к юристу для консультации."

**Формат ответа (JSON):**
```json
{
  "answer": "Ответ с пояснениями...",
  "sources": [
    {
      "text": "Цитируемый фрагмент",
      "reference": "ЖК РФ, ст. 26",
      "url": "https://base.garant.ru/12138291/7/#block_26",
      "confidence": 0.95
    }
  ],
  "metadata": {
    "warning": "Предупреждения о рисках",
    "next_steps": "Рекомендации (например, 'подать заявление в МФЦ')"
  }
}
"""

def dialog(api_key, file, role):
    consult = LegalConsult(api_key, role)
    while True:
        client = input("Клиент:")
        consult_answer = consult.get_answer(client)
        print("Консультант: " + consult_answer, file=file)



if __name__ == '__main__':
    load_dotenv()
    api_key = os.getenv('API_COMPRESSA_KEY')

    with open(f'data/processed/role.txt', 'w') as fl:
        print(role, file=fl)

    with open('data/processed/result.txt', 'w') as fl:
        dialog(api_key, sys.stdout, role)