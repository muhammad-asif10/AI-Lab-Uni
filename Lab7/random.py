import random, os
random_num = random.randint(1, 100)
with open('random_number.txt', 'w', encoding='utf-8') as file:
    file.write(str(random_num))
    print(f'Random number: {random_num}')
    print(f"Random number saved to: {os.path.abspath('random_number.txt')}")