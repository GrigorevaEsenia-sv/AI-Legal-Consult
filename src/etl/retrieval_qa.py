from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from datetime import datetime
# from data_loader import load_data
import sys
from legal_consult import LegalConsult
from dotenv import load_dotenv
import yaml
import os
import uuid

def load_yaml_to_env(yaml_path):
    with open(yaml_path, 'r') as file:
        secrets = yaml.safe_load(file)
    
    os.environ['API_COMPRESSA_KEY'] = secrets['api_keys']['compressa']


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

def dialog(api_key, role):
    consult = LegalConsult(api_key, role)

    # Создаем уникальное имя файла с timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    result_dir = 'data/processed/results/'
    result_filename = f"dialog_{timestamp}_{unique_id}.txt"
    result_path = os.path.join(result_dir, result_filename)
    
    # Создаем директорию если нужно
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    with open(result_path, 'w', encoding='utf-8') as f:
        while True:
            client = input("Клиент: ")
            
            # Записываем вопрос клиента
            f.write(f"Клиент: {client}\n")
            
            if client.lower() in ["спасибо", "благодарю", "завершить"]:
                # Записываем завершение диалога
                footer = f"\n{'='*50}\nКонсультант: Было приятно помочь!\n{'='*50}"
                print(footer.strip())
                f.write(footer)
                break
                
            # Получаем и записываем ответ консультанта
            consult_answer = consult.get_answer(client)
            print(f"Консультант: {consult_answer}")  # Выводим в консоль
            f.write(f"Консультант: {consult_answer}\n\n")  # Записываем в файл



if __name__ == '__main__':
    load_yaml_to_env('configs/llm_config.yaml')
    api_key = os.getenv('API_COMPRESSA_KEY')

    with open(f'data/processed/role.txt', 'w') as fl:
        print(role, file=fl)

    dialog(api_key, role)