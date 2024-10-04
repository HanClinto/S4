# S4
S4 - Super Simple Semantic Search - minimalistic demo of semantic search with BGE-small-en-1.5

## Explanation

This program reads in a datafile in CSV format. It then takes one of the columns (in this case, product description) and generates a semantic embedding for each of these rows. Each row is called a "document", and each embedding is a "vector", which is just a fancy word for a big array of numbers. In this case, the model [bge-small-en-1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) embeds to 384 dimensions, so an array of 384 floating-point numbers.

These embeddings are then stored in a simple array, but if one is doing a larger system, then any vector database will do.

But if you've only got 1000-or-so documents in your search space, there's no need to use a specialized vector database, because brute-force search is plenty fast enough.

The code then embeds a search query that we will encode using the same model, to the same vector space.

The idea is that queries that are semantically similar (similar in meaning) should be numerically similar (comparing the cosine similarity between the numeric embedding vectors).

Cosine similarity ranges from -1 to 1. Values closer to 1 mean that the inputs are more similar, and closer to -1 means that they are more dissimilar. In general, if your embeddings have similarity less than 0.5, then they're not related -- but this can vary from dataset to dataset, so be sure to check for your particular application.

```
from FlagEmbedding import FlagModel
model = FlagModel('BAAI/bge-small-en-v1.5', use_fp16=True)

embedding_1 = model.encode("My printer is printing vertical streaks of toner on the page")
embedding_2 = model.encode("My printer won't turn on.")

embedding_query = model.encode("My pages have up and down lines on them.")

similarity1 = embedding_1 @ embedding_query.T
similarity2 = embedding_2 @ embedding_query.T

if (similarity1 < similarity2):
  print("Search query is closer in meaning to the first document.")
else:
  print("Search query is closer in meaning to the second document.")
```

Calculate the embedding for your query, and then calculate the cosine similarity of the query against each of the documents in your database. A vector database is often used when there is a very large corpus of documents, but for our use-case, we can do a simple brute-force search and simply sort the output by distance, descending. Take the top K results, and return them as search results to the user.

Voila! Simple semantic search!

Results can be improved by adding prompts to the documents and/or search queries, and possibly by [fine-tuning](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) your model. Feel free to open an issue in this repository if you have any questions, or e-mail me directly (hanclinto at gmail).
