# S4 - Super Simple Semantic Search - minimalistic demo of semantic search with BGE-small-en-1.5 

# Requirements:
# pip install FlagEmbedding pandas tqdm

from FlagEmbedding import FlagModel

model = FlagModel('BAAI/bge-small-en-v1.5')

# First, import our data (input.csv)
import pandas as pd
import tqdm as tqdm
import time

data = pd.read_csv("describe3.csv")

# Next, grab our "document" column (what we want to embed) and our "key" column (what we want to be returned on a lookup)
col_document = "Description"
col_key = "ProductID"

col_embedding = "Embedding"

# Now, let's embed our data, saving it to a new "Embedding" column that is our semantic vector, displaying a nice progress bar while we do so.

print(f'Embedding {len(data)} documents...')

embeddings = []
for i, row in tqdm.tqdm(data.iterrows(), total=len(data)):
    embeddings.append(model.encode(row[col_document]))

data[col_embedding] = embeddings

# Optional: Save this data to a new CSV file so that the embeddings don't have to be calculated at every startup.
data.to_csv("input_with_embeddings.csv", index=False)

# Now, let's define a function that will take a query and return the most similar items in our data.
def search(query, top_n=5):
    print(f' Searching for documents related to: {query}')
    start_time = time.time()
    query_embedding = model.encode(query)
    # Distance is calculated as the cosine similarity between the query and the document.
    # similarity = embeddings_1 @ embeddings_2.T
    search_time = time.time()
    query_distances = data[col_embedding].apply(lambda x: query_embedding @ x.T)

    final_time = time.time()

    print(f"  Embedded query in {search_time - start_time}")
    print(f"  Nearest-neighbor search in {final_time - search_time}")
    print(f"  Total time: {final_time - start_time}")

    # Now, let's sort our data by the distance to the query and return the top N results with each of their distances.
    results = data.copy()
    results["query_distance"] = query_distances
    results = results.sort_values("query_distance", ascending=False).head(top_n)

    return results.reset_index()

search_query = "Artwork depicting archery"

while (len(search_query) > 0):
    # Let's test our search function with a query.
    results = search(search_query, top_n=5)

    print(f"Results for query: {search_query}")

    print(results)

    for i, row in results.iterrows():
        # Print the rank, distance, key, and document for each result.
        print(f"Rank: {i}, Distance: {row['query_distance']}, Key: {row[col_key]}, Document: {row[col_document]}")


    print()
    print('What is your query? (blank to exit)')
    search_query = input()

print("All done")







