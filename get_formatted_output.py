from google import genai
import json
import re
# Configure Gemini client once
client = genai.Client(api_key="AIzaSyBjP9HDC2xOukvsBo7QdWuie7GBV49wExM")  # Replace with your actual key

def get_laptop_requirements_from_user_input(user_input: str) -> dict:
    prompt = f"""
You are a helpful assistant that extracts laptop specifications from user descriptions and returns them as a JSON object.

Your goal is to identify and output the values for the following fields in JSON format:
- "use_case": A short label describing the primary use (e.g., "Web Browsing", "Gaming", "Software Development").
- "Min_RAM_GB": The minimum required RAM in GB (as an integer).
- "Min_Storage_GB": The minimum required storage in GB (as an integer).
- "Required_GPU": Specify "Integrated" or "Dedicated".
- "Preferred_CPU": Indicate the preferred CPU type (e.g., "i5", "Ryzen 7").
- "Max_Budget_USD": The maximum budget in USD (as an integer, or null if not mentioned).

Ensure the output is a single, valid JSON object. Do not include any introductory or concluding text. If a specific value isn't clearly stated by the user, use your best judgment to provide a reasonable default based on the 'use_case', or set it to null if it's truly ambiguous (like the budget).

---

Example:

User Input: I need a laptop for coding and running virtual machines. I'd like at least 16GB of RAM and a decent processor, maybe an i7. My budget is around $1200.

JSON Output:
{{
  "use_case": "Software Development",
  "Min_RAM_GB": 16,
  "Min_Storage_GB": 512,
  "Required_GPU": "Integrated",
  "Preferred_CPU": "i7",
  "Max_Budget_USD": 1200
}}

---

Now, process the following user input and generate the JSON output:
User Input: {user_input}
"""

    try:
        response = client.models.generate_content(
                model="gemini-2.0-flash", contents=prompt
            )
        match = re.search(r'\{.*?\}', response.text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in the text.")
        json_str = match.group(0)
        return json.loads(json_str)
    except Exception as e:
        print("❌ Gemini error or parsing issue:", e)
        return {}

# user_input = "I'm a graphic designer using Adobe Photoshop and Illustrator daily. I care a lot about screen quality and performance."
# result = get_laptop_requirements_from_user_input(user_input)
# print(result)
