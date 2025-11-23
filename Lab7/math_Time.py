import math
from datetime import datetime
# Calculate square root
sqrt_value = math.sqrt(16)
print(f"Square root of 16: {sqrt_value}")  # 4.0
 # Use pi constant
circle_area = math.pi * (5 ** 2)
print(f"Area of circle with radius 5: {circle_area:.2f}")  # ~78.54
 # Current date/time and formatting
now = datetime.now()
print(f"Current date and time: {now}")
formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
print(f"Formatted: {formatted_date}")

