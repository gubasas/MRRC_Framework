from transformer import pipeline
pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct", device="mps")  # MPS for M4
response = pipe("Tell me a story about a cat.", max_new_tokens=100)
print(response[0]['generated_text'])