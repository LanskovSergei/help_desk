from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
import os


os.environ["OPENAI_API_KEY"] = ""

documents = SimpleDirectoryReader("./docs").load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

while True:
    query = input("Введите вопрос (или 'exit' для выхода): ")
    if query.lower() in ['exit', 'quit']:
        break
    response = query_engine.query(query)
    print(f"\nОтвет:\n{response}\n")
