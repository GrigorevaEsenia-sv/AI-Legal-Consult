import csv
from typing import List, Dict

def parse_consultations_csv(file_path: str) -> List[Dict[str, str]]:
    """
    Парсит CSV-файл с консультациями и возвращает список словарей.
    Каждый словарь представляет одну консультацию с полями 'Вопрос' и 'Ответ'.
    
    Args:
        file_path (str): Путь к CSV-файлу
        
    Returns:
        List[Dict[str, str]]: Список консультаций
    """
    consultations = []
    
    with open(file_path, mode='r', encoding='utf-8') as csv_file:
        # Используем DictReader для удобства работы с заголовками
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        
        for row in csv_reader:
            # Очищаем данные от лишних пробелов
            question = row.get('Вопрос', '').strip()
            answer = row.get('Ответ', '').strip()
            
            if question and answer:  # Добавляем только если есть и вопрос и ответ
                consultations.append({
                    'Вопрос': question,
                    'Ответ': answer
                })
    
    return consultations

def save_consultations_to_csv(consultations: List[Dict[str, str]], output_path: str):
    """
    Сохраняет список консультаций в CSV-файл.
    
    Args:
        consultations (List[Dict[str, str]]): Список консультаций
        output_path (str): Путь для сохранения CSV-файла
    """
    with open(output_path, mode='w', encoding='utf-8', newline='') as csv_file:
        fieldnames = ['Вопрос', 'Ответ']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        for consultation in consultations:
            writer.writerow(consultation)

def main():
    # Пример использования
    input_csv = 'data/raw/consultations/Q&A_JK.docx'  # Путь к исходному CSV-файлу
    output_csv = 'parsed_consultations.csv'  # Путь для сохранения обработанных данных
    
    try:
        # Парсим консультации
        consultations = parse_consultations_csv(input_csv)
        
        # Сохраняем результат
        save_consultations_to_csv(consultations, output_csv)
        
        print(f"Успешно обработано {len(consultations)} консультаций.")
        print(f"Результат сохранен в файл: {output_csv}")
        
    except FileNotFoundError:
        print(f"Ошибка: файл {input_csv} не найден.")
    except Exception as e:
        print(f"Произошла ошибка при обработке файла: {str(e)}")

if __name__ == '__main__':
    main()