import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import json

class GarantAPILoader:
    def __init__(self, api_key: str):
        self.base_url = "https://api.garant.ru/v1"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def search_documents(self, text: str, count: int = 2, 
                         kind: List[str] = ["001"], 
                         sort: int = 0, sortOrder: int = 0) -> Optional[List[Dict]]:
        """Поиск документов через API ГАРАНТ"""
        url = f"{self.base_url}/search"
        payload = {
            "text": text,
            "count": min(count, 30),  # API ограничивает максимум 30 документов
            "kind": kind,
            "sort": sort,  # 0 - по релевантности
            "sortOrder": sortOrder  # 0 - по убыванию
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json().get("documents", [])
        except requests.exceptions.RequestException as e:
            print(f"Ошибка поиска документов: {str(e)}")
            return None

    def export_html(self, topic_id: int) -> Optional[str]:
        """Экспорт документа в HTML формате"""
        url = f"{self.base_url}/topic/{topic_id}/html"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            html_data = response.json()
            
            # Обработка HTML страниц
            full_html = []
            for page in html_data.get("items", []):
                soup = BeautifulSoup(page["text"], 'html.parser')
                
                # Удаление ненужных элементов
                for elem in soup.find_all(class_=["comment", "ad", "hidden"]):
                    elem.decompose()
                
                full_html.append(str(soup))
            
            return "\n".join(full_html)
        except requests.exceptions.RequestException as e:
            print(f"Ошибка экспорта документа {topic_id}: {str(e)}")
            return None

    def process_search_results(self, query) -> List[Dict]:
        """Полный процесс: поиск + экспорт"""
        documents = self.search_documents(**query)
        if not documents:
            return []

        results = []
        for doc in documents:  # Берем первые 2 документа
            html_content = self.export_html(doc["topic"])
            if html_content:
                results.append({
                    "title": doc["name"],
                    "url": f"https://api.garant.ru/v1/topic/{doc['topic']}/html",
                    "topic_id": doc["topic"],
                    "html": html_content  # Для примера показываем часть контента
                })
        
        return results

# Пример использования
if __name__ == "__main__":
    API_KEY = "003ac54e243711f095560050568d72f0"  # Замените на ваш действительный токен
    
    loader = GarantAPILoader(api_key=API_KEY)
    
    # Параметры поиска (можно менять)
    search_params = {
        "text": "перепланировка квартиры",
        "count": 2,
        "kind": ["001"],  # Федеральное законодательство
        "sort": 0,  # По релевантности
        "sortOrder": 0  # По убыванию
    }
    
    results = loader.process_search_results(search_params)

    print(f"Найдено документов: {len(results)}")
    for idx, doc in enumerate(results, 1):
        with open(f'data/raw/housing_code/garant/{idx}.html', 'w') as f:
            f.write(doc['html'])
        print(f"Ссылка: {doc['url']}")
        print(f"\nДокумент #{idx}:")
        print(f"HTML: {doc['html'][:200]}...")