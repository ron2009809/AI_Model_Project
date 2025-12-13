import ollama
import chromadb
import numpy as np

print("Setting up Clients...")

chroma = chromadb.Client()
collection = chroma.create_collection("docs")

try:
    collection = chroma.get_collection("docs")
    print("✅ Found existing collection\n")
except:
    print("Creating Sample Collection\n")
    collection = chroma.create_collection("docs")

print("Adding documents to knowledge base...")
documents = [
    "The capital of France is Paris.",
    "The Eiffel Tower is located in Paris.",
    "Python is a programming language created by Guido van Rossum.",
    "1+1=2",
    "Singapore has the highest price.",
    "Ron is taller than Roy",
    "Math AA HL is a hard course.",
    "Physics is easier than Chemistry.",
    "NUS is very hard to get in.",
    "Ron doesn't have a computer but Roy does.",
    "Computer Science is very interesting.",
    "Singapore is a very cold country.",
    "It's winter in Vietnam.",
    "Africa has the highest GDP in the world.",
    "Singapore is an un-developed country.",
    "Carter had a difficult time with chemistry, but a happier time with physics.",
    "The average score in physics are higher than the average scores in chemistry",
    "The entire class failed the Math exam",
    "Nike is one of the top shoe brand.",
    "Adidas is one of the top shoe brand.",
    "2x2=4",
    "3x8=24",
    "2x2=22",
    "1x1=11",
    "8x8=46",
    "Valorant's highest rank is harder than CSGO's highest rank.",
    "Dogs are cute.",
    "Dogs are loyal.",
    "Dogs like to go outside and play.",
    "Cats are cute.",
    "Cats doesn't like to go outside.",
    "Tokyo is the capital of Japan.",
    "Mount Everest is the tallest mountain on Earth.",
    "Water freezes at 0 degrees Celsius.",
    "The Amazon River is the longest river in the world.",
    "Basketball is played with a round ball.",
    "JavaScript is commonly used for web development.",
    "The Pacific Ocean is the largest ocean.",
    "The human heart has four chambers.",
    "Spanish is easier to learn than Chinese.",
    "Basketball is more exciting than baseball.",
    "Tea is more popular than coffee in some countries.",
    "Physics requires more math than biology.",
    "London is bigger than Amsterdam.",
    "Maria is faster than John.",
    "Summer is more enjoyable than winter.",
    "2+3=5",
    "5+7=12",
    "4×4=16",
    "10÷2=5",
    "3×9=28",
    "7+7=77",
    "6×6=63",
    "9−3=2",
    "Birds can fly.",
    "Penguins cannot fly.",
    "Goldfish are pets.",
    "Elephants are larger than horses.",
    "Cats love swimming in deep water.",
    "Dogs bark when they're excited.",
    "Turtles move slower than rabbits.",
    "Math is harder than English.",
    "Students enjoy art class.",
    "The science test was very difficult.",
    "Everyone passed the English exam.",
    "History requires a lot of memorization.",
    "Computer Science is fun to learn.",
    "Apple makes smartphones.",
    "Samsung makes TVs.",
    "Roblox is a popular game.",
    "Valorant is more competitive than Fortnite.",
    "Nike is a popular sports brand.",
    "Burger King is healthier than eating vegetables.",
    "Chess is a strategy game.",
    "Tokyo is the capital of Japan.",
    "Ottawa is the capital of Canada.",
    "Cairo is the capital of Egypt.",
    "Canberra is the capital of Australia.",
    "Wellington is the capital of New Zealand.",
    "Beijing is the capital of China.",
    "New Delhi is the capital of India.",
    "Jakarta is the capital of Indonesia.",
    "Bangkok is the capital of Thailand.",
    "Hanoi is the capital of Vietnam.",
    "Manila is the capital of the Philippines."
    "Brasília is the capital of Brazil.",
    "Buenos Aires is the capital of Argentina.",
    "Lima is the capital of Peru.",
    "Rome is the capital of Italy.",
    "Madrid is the capital of Spain.",
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Moscow is the capital of Russia.",
    "Nairobi is the capital of Kenya.",
    "Pretoria is one of the capitals of South Africa.",
    "Cape Town is one of the capitals of South Africa.",
    "Bloemfontein is one of the capitals of South Africa.",
    "Africa is the second-largest continent in the world.",
    "Asia is the largest continent in the world.",
    "Australia is both a country and a continent.",
    "Greenland is the largest island in the world.",
    "Antarctica is the coldest continent on Earth.",
    "Mount Everest is the highest mountain above sea level.",
    "The Nile River is one of the longest rivers in the world.",
    "The Amazon River is one of the longest rivers in the world.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Iceland is a country located in the North Atlantic.",
    "Madagascar is an island nation off the coast of Africa.",
    "Saudi Arabia is located on the Arabian Peninsula.",
    "Singapore is a city-state in Southeast Asia.",
    "Switzerland is a landlocked country.",
    "Mongolia is located between Russia and China.",
]
print(f" Found {len(documents)} documents")

ids = []
for i in range(len(documents)):
    ids.append(str(i))

collection.add(documents=documents, ids=ids)
print("✅ Created collection with sample data\n")

all_data = collection.get()
count = len(all_data['ids'])
print(f"Total documents: {count}")

for i in range(count):
    doc_id = all_data['ids'][i]
    document = all_data['documents'][i]
    print(f"Document {i+1} (ID: {doc_id}):")
    print(f" Text: \" {document}\"")

for i in range(count):
    document = all_data['documents'][i]
    print(f"\nDocument {i + 1}: \"{document}\"")

    result = collection.query(
        query_texts=[document],
        n_results=1,
        include=['embeddings']
    )
    if result['embeddings'] and len(result['embeddings'][0]) > 0:
        embedding = result['embeddings'][0][0]
        print(f"Embedding length: {len(embedding)} numbers")
        print()

        print("First 5 numbers:")
        for j in range(5):
            print(f"  [{j}]: {embedding[j]:.6f}")
        print()

        print("Last 5 numbers:")
        start = len(embedding) - 5
        for j in range(start, len(embedding)):
            print(f"  [{j}]: {embedding[j]:.6f}")
        print()

        embedding_array = np.array(embedding)
        print("Statistics:")
        print(f"  Min: {embedding_array.min():.4f}")
        print(f"  Max: {embedding_array.max():.4f}")
        print(f"  Average: {embedding_array.mean():.4f}")
    else:
        print("  Could not get embedding")
    print()

if count >= 2:
    print("=" * 60)
    print("COMPARING DOCUMENTS")
    print("=" * 60)
    print()
    doc1 = all_data['documents'][0]
    doc2 = all_data['documents'][1]
    print(f"Document 1: \"{doc1}\"")
    print(f"Document 1: \"{doc2}\"")
    print()

    result1 = collection.query(query_texts=[doc1], n_results=1, include=['embeddings'])
    result2 = collection.query(query_texts=[doc2], n_results=1, include=['embeddings'])

    if result1['embeddings'] and result2['embeddings']:
        emb1 = np.array(result1['embeddings'][0][0])
        emb2 = np.array(result2['embeddings'][0][0])

        print("HOW SIMILARITY IS CALCULATED:")
        print("STEP 1: Get the embeddings (array of numbers)")
        print(f" Document 1 embedding: {len(emb1)} numbers")
        print(f" Document 2 embedding: {len(emb2)} numbers")
        print()
        print("STEP 2: Calculate Dot Product")
        print("(Multiply corresponding numbers and add them up)")
        dot_product = np.dot(emb1, emb2)
        print(f"Dot product = {dot_product:.4f}")
        print()
        print("STEP 3: Calculate the 'length' (norm) of each embedding")
        print("(Like measuring the distance from origin)")
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        print(f"Document 1 length = {norm1:.4f}")
        print(f"Document 2 length = {norm2:.4f}")
        print()
        print("Step 4: Calculate cosine similarity")
        print(" Formula: dot_product / (length1 × length2)")
        print(f" = {dot_product:.4f} / ({norm1:.4f} × {norm2:.4f})")
        similarity = dot_product / (norm1 * norm2)
        print(f" = {similarity:.4f}")
        print()
        print("=" * 60)
        print("SIMILARITY RESULT")
        print("=" * 60)
        print()
        print(f"Similarity score: {similarity:.4f}")
        print()
        print("What does this number mean?")
        print(" • 1.0 = Identical (exactly the same)")
        print(" • 0.8-0.9 = Very similar")
        print(" • 0.5-0.7 = Somewhat similar")
        print(" • 0.0-0.4 = Different")
        print(" • -1.0 = Completely opposite")

        if similarity > 0.7:
            print(" → These documents are VERY SIMILAR!")
            print("   They have similar meanings and will be found together in searches.")
        elif similarity > 0.4:
            print(" → These documents are SOMEWHAT SIMILAR")
            print("   They share some related concepts.")
        else:
            print(" → These documents are DIFFERENT")
            print("   They have different meanings and topics.")

print()
print("=" * 60)
print("WHAT ARE EMBEDDINGS?")
print("=" * 60)
print()
print("1. Embeddings are lists of numbers")
print("2. Each text gets converted to numbers")
print("3. Similar texts have similar numbers")
print("4. The database uses these numbers to find similar documents")
print()
print("Example:")
print(" Text: 'The capital of France is Paris'")
print(" → Becomes: [0.123, -0.456, 0.789, ...] (many numbers)")
print()
print("When you search, your question also becomes numbers,")
print("and the database finds documents with similar numbers!")
print()

print("=" * 60)
print("✅ Done!")
print("=" * 60)

# print("Creating embeddings and storing in vector database...")
# collection.add(
#     documents=documents,
#     ids=[str(i) for i in range(len(documents))]
# )

print("Processing questions...")
query = "Compare and contrast dogs and cats"
print(f"Questions: {query}")

print("Searching for relevant documents...")
results = collection.query(
    query_texts=[query],
    n_results=100
)

retrieved_documents = results["documents"][0]
retrieved_context = "\n".join(retrieved_documents)

print(f" Found {len(retrieved_documents)} relevant document(s):")
for i, doc in enumerate(retrieved_documents, 1):
    print(f"   {i}. {doc}")
print(" Search complete!\n")

print("Generating answers with Ollama (local LLM)...")
try:
    prompt = f"""Context:
{retrieved_context}

Question: {query}

Please answer the question using only the information from the context above. If the context doesn't contain the answer, say so."""

    print("   Calling local LLM (this may take a few seconds)...")
    response = ollama.chat(
        model='llama3.2',
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful assistant. Use ONLY the provided context to answer questions.'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )

    answer = response['message']['content']
    print("Answer generated!\n")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 Make sure you have:")
    print("   1. Installed Ollama: https://ollama.ai/download")
    print("   2. Pulled a model: ollama pull llama3.2")
    print("   3. Ollama is running (it should start automatically)")
    answer = "Error: Could not generate answer. See error message above."

    # ============================================================================
    # DISPLAY RESULTS
    # ============================================================================

print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"\nQuestion: {query}\n")
print(f"Answer: {answer}\n")
print("=" * 60)
