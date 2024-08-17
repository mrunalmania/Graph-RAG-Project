import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from langchain_community.graphs import Neo4jGraph
import os
import pandas as pd
import tiktoken  # Use tiktoken for tokenization

os.environ["NEO4J_URI"] = "bolt://44.202.90.6"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "puffs-cruise-boys"

graph = Neo4jGraph(refresh_schema=False)

# We are going to use the news press release api to extract the useful data for our project.

news  = pd.read_csv(
    "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
)

# Get the encoding from tiktoken
encoding = tiktoken.get_encoding("cl100k_base")

# Calculate the number of tokens for each row
news["tokens"] = [
    len(encoding.encode(f"{row['title']} {row['text']}"))
    for i, row in news.iterrows()
]

print(news.head())


