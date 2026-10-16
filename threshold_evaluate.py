import re
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DOCUMENTS_DIR = "documents"

IN_DOMAIN = [
    "What is food hygiene according to Codex?",
    "What are Good Hygiene Practices (GHPs)?",
    "Why is training important in a food hygiene system?",
    "What should cleaning accomplish?",
    "What is the requirement for petty food business registration?",
    "What are the basic requirements for a petty food manufacturer?",
    "What are the limits for aflatoxin and aflatoxin M1?",
    "What is the mercury limit in fish?",
    "What is meant by frozen dessert or frozen confection?",
    "What are the microbiological requirements for food products?",
    "Why should food workers receive hygiene training?",
    "How does cleaning food-contact surfaces prevent contamination?",
    "How can contamination between raw and ready-to-eat food be reduced?",
    "What must a small food business do before operating?",
    "What happens if contaminants exceed the specified FSSAI limits?",
    "What are naturally occurring toxic substances and their limits?",
    "What food products are covered by frozen dessert standards?",
    "Why are food additives mentioned in the regulations?",
    "What is a prerequisite programme?",
    "What is a significant hazard?",
    "What is HACCP?",
    "What is a Registering Authority?",
    "What is a State Licensing Authority?",
    "What is aflatoxin M1 and what is its limit in milk?",
    "What is the cadmium limit in turmeric?",
    "What is methyl mercury?",
    "What are Codex hygiene requirements and FSSAI food-business requirements?",
    "How are contamination prevention requirements different from contaminant limits?",
    "What are the hygiene responsibilities of a food business and how do contaminant limits relate to them?",
    "How do hygiene practices relate to food product standards?",
    "Why is food hygiene training necessary?",
    "How should raw food and ready-to-eat food be separated?",
    "What is the difference between cleaning and disinfection?",
    "What are the registration and hygiene obligations of a small food business?",
    "Can a food product be considered safe if it exceeds a contaminant limit?",
]

OUT_OF_DOMAIN = [
    "Who is the Prime Minister of India?",
    "What is the capital of France?",
    "How do I write a Python program for a calculator?",
    "What is the weather today?",
    "Who won the FIFA World Cup?",
    "How do I calculate the area of a triangle?",
    "What is the latest iPhone model?",
    "Explain Newton's second law of motion.",
    "What is the population of India?",
    "How do I make a website using HTML and CSS?",
]

def create_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

print("Loading documents...")
documents = {}

for filename in [
    "cc6125en.pdf",
    "Contaminants_Regulations.pdf",
    "Food_Additives_Regulations.pdf",
    "Licensing_Regulations.pdf",
]:
    reader = PdfReader(f"{DOCUMENTS_DIR}/{filename}")
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    documents[filename] = text

all_chunks = []
for filename, text in documents.items():
    for i, chunk in enumerate(create_chunks(text)):
        all_chunks.append({
            "document": filename,
            "chunk_id": i,
            "text": chunk
        })

print(f"Knowledge chunks: {len(all_chunks)}")

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [x["text"] for x in all_chunks]

embeddings = model.encode(
    texts,
    show_progress_bar=True
)

def best_semantic_score(question):
    q_embedding = model.encode([question])
    scores = cosine_similarity(q_embedding, embeddings)[0]
    return float(np.max(scores))

print("\nCalculating best semantic similarity scores...")
in_scores = [(q, best_semantic_score(q)) for q in IN_DOMAIN]
out_scores = [(q, best_semantic_score(q)) for q in OUT_OF_DOMAIN]

thresholds = [0.25, 0.30, 0.35, 0.40, 0.45]

print("\n" + "=" * 78)
print("RELEVANCE THRESHOLD EXPERIMENT")
print("=" * 78)

print("\nThreshold interpretation:")
print("score >= threshold -> ACCEPT as relevant")
print("score < threshold  -> REJECT as insufficient context")

print("\n" + "-" * 78)
print(f"{'Threshold':<12}{'In-domain accepted':<22}{'In-domain rejected':<22}{'OOD rejected':<15}")
print("-" * 78)

for threshold in thresholds:
    in_accept = sum(score >= threshold for _, score in in_scores)
    in_reject = len(IN_DOMAIN) - in_accept
    out_reject = sum(score < threshold for _, score in out_scores)

    print(
        f"{threshold:<12.2f}"
        f"{in_accept}/{len(IN_DOMAIN):<21}"
        f"{in_reject}/{len(IN_DOMAIN):<21}"
        f"{out_reject}/{len(OUT_OF_DOMAIN)}"
    )

print("\n" + "=" * 78)
print("DETAILED SCORES")
print("=" * 78)

print("\nIN-DOMAIN QUESTIONS")
for q, score in sorted(in_scores, key=lambda x: x[1]):
    print(f"{score:.4f}  |  {q}")

print("\nOUT-OF-DOMAIN QUESTIONS")
for q, score in sorted(out_scores, key=lambda x: x[1], reverse=True):
    print(f"{score:.4f}  |  {q}")

print("\n" + "=" * 78)
print("RECOMMENDATION GUIDE")
print("=" * 78)
print("Choose the threshold that keeps in-domain rejection low while")
print("rejecting as many unrelated questions as possible.")
print("This is a small evaluation set, so the selected threshold is a")
print("starting point and should be re-tested when the document collection changes.")
