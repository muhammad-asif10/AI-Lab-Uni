import mymodule

print(mymodule.greet("Student"))  # Output: Hello, Student! Welcome to Week 8.
result = mymodule.add_numbers(5, 10)
print(f"The sum is: {result}")    

from mymodule import greet, add_numbers
print(greet("Student"))
result = add_numbers(5, 10)
print(f"The sum is: {result}")

from mymodule import multiply_numbers
result_1 = multiply_numbers(5,7)
print(f"The Multiplication is: {result_1}")