from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

response=client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role":"user","content":"What is kubernets?"}
    ]
)
print(response.choices[0].message.content)