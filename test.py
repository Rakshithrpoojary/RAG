from openai import OpenAI

client = OpenAI(api_key="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")

response=client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role":"user","content":"What is kubernets?"}
    ]
)
print(response.choices[0].message.content)