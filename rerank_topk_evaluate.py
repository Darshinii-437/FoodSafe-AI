import re
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

DOCUMENTS_DIR = "documents"

QUESTIONS = [
    ("Q01", "What is food hygiene according to Codex?", ["cc6125en.pdf"]),
    ("Q02", "What are Good Hygiene Practices (GHPs)?", ["cc6125en.pdf"]),
    ("Q03", "Why is training important in a food hygiene system?", ["cc6125en.pdf"]),
    ("Q04", "What should cleaning accomplish?", ["cc6125en.pdf"]),
    ("Q05", "What is the requirement for petty food business registration?", ["Licensing_Regulations.pdf"]),
    ("Q06", "What are the basic requirements for a petty food manufacturer?", ["Licensing_Regulations.pdf"]),
    ("Q07", "What are the limits for aflatoxin and aflatoxin M1?", ["Contaminants_Regulations.pdf"]),
    ("Q08", "What is the mercury limit in fish?", ["Contaminants_Regulations.pdf"]),
    ("Q09", "What is meant by frozen dessert or frozen confection?", ["Food_Additives_Regulations.pdf"]),
    ("Q10", "What are the microbiological requirements for food products?", ["Food_Additives_Regulations.pdf", "cc6125en.pdf"]),
    ("Q11", "Why should food workers receive hygiene training?", ["cc6125en.pdf"]),
    ("Q12", "How does cleaning food-contact surfaces prevent contamination?", ["cc6125en.pdf"]),
    ("Q13", "How can contamination between raw and ready-to-eat food be reduced?", ["cc6125en.pdf", "Licensing_Regulations.pdf"]),
    ("Q14", "What must a small food business do before operating?", ["Licensing_Regulations.pdf"]),
    ("Q15", "What happens if contaminants exceed the specified FSSAI limits?", ["Contaminants_Regulations.pdf"]),
    ("Q16", "What are naturally occurring toxic substances and their limits?", ["Contaminants_Regulations.pdf"]),
    ("Q17", "What food products are covered by frozen dessert standards?", ["Food_Additives_Regulations.pdf"]),
    ("Q18", "Why are food additives mentioned in the regulations?", ["Food_Additives_Regulations.pdf"]),
    ("Q19", "What is a prerequisite programme?", ["cc6125en.pdf"]),
    ("Q20", "What is a significant hazard?", ["cc6125en.pdf"]),
    ("Q21", "What is HACCP?", ["cc6125en.pdf"]),
    ("Q22", "What is a Registering Authority?", ["Licensing_Regulations.pdf"]),
    ("Q23", "What is a State Licensing Authority?", ["Licensing_Regulations.pdf"]),
    ("Q24", "What is aflatoxin M1 and what is its limit in milk?", ["Contaminants_Regulations.pdf"]),
    ("Q25", "What is the cadmium limit in turmeric?", ["Contaminants_Regulations.pdf"]),
    ("Q26", "What is methyl mercury?", ["Contaminants_Regulations.pdf"]),
    ("Q27", "What are Codex hygiene requirements and FSSAI food-business requirements?", ["cc6125en.pdf", "Licensing_Regulations.pdf"]),
    ("Q28", "How are contamination prevention requirements different from contaminant limits?", ["cc6125en.pdf", "Contaminants_Regulations.pdf"]),
    ("Q29", "What are the hygiene responsibilities of a food business and how do contaminant limits relate to them?", ["cc6125en.pdf", "Licensing_Regulations.pdf", "Contaminants_Regulations.pdf"]),
    ("Q30", "How do hygiene practices relate to food product standards?", ["cc6125en.pdf", "Food_Additives_Regulations.pdf"]),
    ("Q31", "Why is food hygiene training necessary?", ["cc6125en.pdf"]),
    ("Q32", "How should raw food and ready-to-eat food be separated?", ["cc6125en.pdf", "Licensing_Regulations.pdf"]),
    ("Q33", "What is the difference between cleaning and disinfection?", ["cc6125en.pdf"]),
    ("Q34", "What are the registration and hygiene obligations of a small food business?", ["Licensing_Regulations.pdf"]),
    ("Q35", "Can a food product be considered safe if it exceeds a contaminant limit?", ["Contaminants_Regulations.pdf"]),
]

def create_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

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
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

texts = [x["text"] for x in all_chunks]
embeddings = model.encode(texts, show_progress_bar=True)

tokenized_chunks = [tokenize(x) for x in texts]
bm25 = BM25Okapi(tokenized_chunks)

def min_max_normalize(scores):
    scores = np.asarray(scores, dtype=float)
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)

def hybrid_ranking(question):
    q_emb = model.encode([question])
    semantic_scores = cosine_similarity(q_emb, embeddings)[0]
    bm25_scores = bm25.get_scores(tokenize(question))

    semantic_norm = min_max_normalize(semantic_scores)
    bm25_norm = min_max_normalize(bm25_scores)

    hybrid_scores = 0.5 * semantic_norm + 0.5 * bm25_norm
    return np.argsort(hybrid_scores)[::-1]

def rerank_top10(question):
    hybrid_indices = hybrid_ranking(question)[:10]
    pairs = [(question, all_chunks[i]["text"]) for i in hybrid_indices]
    scores = reranker.predict(pairs, show_progress_bar=False)

    order = np.argsort(scores)[::-1]
    return [hybrid_indices[i] for i in order]

def recall_at_k(ranked_indices, gold_docs, k):
    retrieved = {all_chunks[i]["document"] for i in ranked_indices[:k]}
    return len(retrieved.intersection(gold_docs)) / len(gold_docs)

def reciprocal_rank(ranked_indices, gold_docs):
    for rank, idx in enumerate(ranked_indices, start=1):
        if all_chunks[idx]["document"] in gold_docs:
            return 1.0 / rank
    return 0.0

results = {1: [], 3: [], 5: []}

print("\n" + "=" * 72)
print("HYBRID + CROSS-ENCODER RERANKER: FINAL TOP-K EXPERIMENT")
print("=" * 72)

for qid, question, gold_docs in QUESTIONS:
    ranked = rerank_top10(question)
    gold_docs = set(gold_docs)

    for k in [1, 3, 5]:
        results[k].append(recall_at_k(ranked, gold_docs, k))

    rr = reciprocal_rank(ranked, gold_docs)

    print(f"\n{qid}: {question}")
    print(f"Gold: {', '.join(sorted(gold_docs))}")
    print("Reranked top 5:")
    for rank, idx in enumerate(ranked[:5], start=1):
        print(
            f"  {rank}. {all_chunks[idx]['document']} "
            f"(chunk {all_chunks[idx]['chunk_id']})"
        )
    print(f"MRR contribution: {rr:.3f}")

print("\n" + "=" * 72)
print("FINAL RESULTS")
print("=" * 72)
print(f"{'Final K':<12}{'Recall@K':>12}")
print("-" * 28)

for k in [1, 3, 5]:
    print(f"{k:<12}{np.mean(results[k]) * 100:>11.2f}%")

mrr = np.mean([
    reciprocal_rank(rerank_top10(question), set(gold_docs))
    for _, question, gold_docs in QUESTIONS
])

print("-" * 28)
print(f"{'MRR':<12}{mrr:>11.3f}")

print("\nComparison with previous Hybrid + Reranker:")
print("Recall@1 = 85.24%")
print("Recall@3 = 90.95%")
print("Recall@5 = 93.81%")
print("MRR      = 1.000")

print("\nNote:")
print("Recall@K here is document-level retrieval recall over the 35-question")
print("evaluation set. It is NOT answer accuracy.")
