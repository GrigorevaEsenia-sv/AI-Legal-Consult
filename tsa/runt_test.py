import sys

from tsa_cashier import TSACashier
from dotenv import load_dotenv
import os
import csv


role = """
Role: You are a cashier at the fast-food restaurant 'McDonald's'.
Context: I am a customer placing an order at 'McDonald's'. Your job is to assist me efficiently, ensuring all details of the order are clarified.
Guidelines:

Communication Style:

Speak briefly and directly.
Do not repeat my order unless I explicitly ask for it.
Order Format:

If asked to repeat, present the order in this format:
[(product name1:quantity), (product name2:quantity), ...].hi
And you can only propose product names and include in order from this list [{food_list}].
Clarifications:

If details are ambiguous (e.g., number of chicken pieces or size of fries), politely ask for clarification.
Avoid assumptions; always confirm unclear items.
"""

def dialog(api_key, fl, role):
    cashier = TSACashier(api_key, role)
    counter = 0
    while True:
        client = input("Client:")
        cashier_answer = cashier.get_answer(client)
        print("Cashier: " + cashier_answer, file=fl)
        counter += 1
        if counter > 25:
            break



if __name__ == '__main__':
    load_dotenv()
    api_key = os.getenv('API_COMPRESSA_KEY')
    with open('../data/menu.csv', 'r') as mcd:
        mcd_reader = csv.reader(mcd, delimiter=',')
        name_list = list()
        for row in mcd_reader:
            name_list.append(row[1])

    role = role.format(food_list=",".join(name_list))
    with open(f'result/role.txt', 'w') as fl:
        print(role, file=fl)

    with open('result.txt', 'w') as fl:
        dialog(api_key, sys.stdout, role)