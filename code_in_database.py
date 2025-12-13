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
    # File Operations
    """# Read a text file
with open('data.txt', 'r') as file:
content = file.read()
print(content)""",

    """# Write to a text file
with open('output.txt', 'w') as file:
file.write('Hello, World!')""",

    """# Read file line by line
with open('data.txt', 'r') as file:
for line in file:
    print(line.strip())""",

    # List Operations
    """# Sort a list of numbers
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_numbers = sorted(numbers)
print(sorted_numbers)""",

    """# Filter a list (get even numbers)
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = [n for n in numbers if n % 2 == 0]
print(even_numbers)""",

    """# Calculate the sum of a list
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)
print(f"Sum: {total}")""",

    """# Find the maximum value in a list
numbers = [3, 7, 2, 9, 1, 5]
max_value = max(numbers)
print(f"Maximum: {max_value}")""",

    """# Calculate the average of a list
numbers = [10, 20, 30, 40, 50]
average = sum(numbers) / len(numbers)
print(f"Average: {average}")""",

    # String Operations
    """# Split a string into words
text = "Hello World Python Programming"
words = text.split()
print(words)""",

    """# Join a list of strings
words = ['Hello', 'World', 'Python']
sentence = ' '.join(words)
print(sentence)""",

    """# Format a string with variables
name = "Alice"
age = 30
message = f"My name is {name} and I am {age} years old"
print(message)""",

    # Dictionary Operations
    """# Access dictionary values
person = {'name': 'John', 'age': 25, 'city': 'New York'}
print(person['name'])
print(person.get('age', 0))""",

    """# Update dictionary values
person = {'name': 'John', 'age': 25}
person['age'] = 26
person['city'] = 'Boston'
print(person)""",

    # Error Handling
    """# Try-except error handling
try:
result = 10 / 0
except ZeroDivisionError:
print("Cannot divide by zero!")""",

    """# Try-except with multiple exceptions
try:
number = int(input("Enter a number: "))
result = 100 / number
print(f"Result: {result}")
except ValueError:
print("Please enter a valid number!")
except ZeroDivisionError:
print("Cannot divide by zero!")""",

    # Loops
    """# For loop through a list
fruits = ['apple', 'banana', 'orange']
for fruit in fruits:
print(f"I like {fruit}")""",

    """# While loop with counter
count = 0
while count < 5:
print(f"Count: {count}")
count += 1""",

    # Functions
    """# Define a function with parameters
def greet(name):
return f"Hello, {name}!"

message = greet("Alice")
print(message)""",

    """# Function with multiple parameters
def calculate_area(length, width):
area = length * width
return area

result = calculate_area(5, 3)
print(f"Area: {result}")""",

    """# Function with default parameters
def greet(name, greeting="Hello"):
return f"{greeting}, {name}!"

print(greet("Bob"))
print(greet("Alice", "Hi"))"""
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
query = "Write a for loop that lists only odd numbers from 0-100"
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
