from ollama import embed, chat

embeddings = embed(model = "nomic-embed-text", input = ["Here is an example of a custom embedding.", "This is another example of a custom embedding.",
"Chickens cross roads to get to the post office"])

print(len(embeddings['embeddings'][0]))

response = chat(model = "phi4-mini", messages = [
    {"role": "user", 
    "content": "Why did the chicken cross the road?"}
    ])
print(response['message']['content'])