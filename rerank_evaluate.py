import os
import re
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENT_FOLDER = "documents"

PDF_FILES = [
    "cc6125en.pdf",
    "Contaminants_Regulations.pdf",
    "Food_Additives_Regulations.pdf",
    "Licensing_Regulations.pdf"
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Hybrid retrieves candidates first.
# Cross-Encoder then reranks them.
CANDIDATE_K = 10

FINAL_K_VALUES = [1, 3, 5]


# ============================================================
# 35 QUESTIONS
# ============================================================

questions = [

    ("Q01", "What is food hygiene according to Codex?",
     ["cc6125en.pdf"]),

    ("Q02", "What are Good Hygiene Practices (GHPs)?",
     ["cc6125en.pdf"]),

    ("Q03", "Why is training important in a food hygiene system?",
     ["cc6125en.pdf"]),

    ("Q04", "What should cleaning accomplish?",
     ["cc6125en.pdf", "Licensing_Regulations.pdf"]),

    ("Q05", "What is the requirement for petty food business registration?",
     ["Licensing_Regulations.pdf"]),

    ("Q06", "What are the basic requirements for a petty food manufacturer?",
     ["Licensing_Regulations.pdf"]),

    ("Q07", "What are the limits for aflatoxin and aflatoxin M1?",
     ["Contaminants_Regulations.pdf"]),

    ("Q08", "What is the mercury limit in fish?",
     ["Contaminants_Regulations.pdf"]),

    ("Q09", "What is meant by frozen dessert or frozen confection?",
     ["Food_Additives_Regulations.pdf"]),

    ("Q10", "What are the microbiological requirements for food products?",
     ["Food_Additives_Regulations.pdf", "cc6125en.pdf"]),

    ("Q11", "Why should food workers receive hygiene training?",
     ["cc6125en.pdf"]),

    ("Q12", "How does cleaning food-contact surfaces prevent contamination?",
     ["cc6125en.pdf", "Licensing_Regulations.pdf"]),

    ("Q13", "How can contamination between raw and ready-to-eat food be reduced?",
     ["cc6125en.pdf", "Licensing_Regulations.pdf"]),

    ("Q14", "What must a small food business do before operating?",
     ["Licensing_Regulations.pdf"]),

    ("Q15", "What happens if contaminants exceed the specified FSSAI limits?",
     ["Contaminants_Regulations.pdf"]),

    ("Q16", "What are naturally occurring toxic substances and their limits?",
     ["Contaminants_Regulations.pdf"]),

    ("Q17", "What food products are covered by frozen dessert standards?",
     ["Food_Additives_Regulations.pdf"]),

    ("Q18", "Why are food additives mentioned in the regulations?",
     ["Food_Additives_Regulations.pdf"]),

    ("Q19", "What is a prerequisite programme?",
     ["cc6125en.pdf"]),

    ("Q20", "What is a significant hazard?",
     ["cc6125en.pdf"]),

    ("Q21", "What is HACCP?",
     ["cc6125en.pdf"]),

    ("Q22", "What is a Registering Authority?",
     ["Licensing_Regulations.pdf"]),

    ("Q23", "What is a State Licensing Authority?",
     ["Licensing_Regulations.pdf"]),

    ("Q24", "What is aflatoxin M1 and what is its limit in milk?",
     ["Contaminants_Regulations.pdf"]),

    ("Q25", "What is the cadmium limit in turmeric?",
     ["Contaminants_Regulations.pdf"]),

    ("Q26", "What is methyl mercury?",
     ["Contaminants_Regulations.pdf"]),

    ("Q27", "What are Codex hygiene requirements and FSSAI food-business requirements?",
     ["cc6125en.pdf", "Licensing_Regulations.pdf"]),

    ("Q28", "How are contamination prevention requirements different from contaminant limits?",
     ["cc6125en.pdf", "Contaminants_Regulations.pdf"]),

    ("Q29", "What are the hygiene responsibilities of a food business and how do contaminant limits relate to them?",
     ["cc6125en.pdf", "Licensing_Regulations.pdf", "Contaminants_Regulations.pdf"]),

    ("Q30", "How do hygiene practices relate to food product standards?",
     ["cc6125en.pdf", "Food_Additives_Regulations.pdf"]),

    ("Q31", "Why is food hygiene training necessary?",
     ["cc6125en.pdf"]),

    ("Q32", "How should raw food and ready-to-eat food be separated?",
     ["cc6125en.pdf", "Licensing_Regulations.pdf"]),

    ("Q33", "What is the difference between cleaning and disinfection?",
     ["cc6125en.pdf", "Licensing_Regulations.pdf"]),

    ("Q34", "What are the registration and hygiene obligations of a small food business?",
     ["Licensing_Regulations.pdf"]),

    ("Q35", "Can a food product be considered safe if it exceeds a contaminant limit?",
     ["Contaminants_Regulations.pdf"])
]


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():

    documents = {}

    print("\nLoading documents...")

    for filename in PDF_FILES:

        path = os.path.join(
            DOCUMENT_FOLDER,
            filename
        )

        reader = PdfReader(path)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        documents[filename] = text

        print(
            f"{filename}: "
            f"{len(reader.pages)} pages"
        )

    return documents


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(text):

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(chunk)

        start = end - CHUNK_OVERLAP

    return chunks


def build_knowledge_base(documents):

    chunks = []

    for filename, text in documents.items():

        document_chunks = create_chunks(text)

        for i, chunk in enumerate(document_chunks):

            chunks.append({
                "document": filename,
                "chunk_id": i,
                "text": chunk
            })

    return chunks


# ============================================================
# LOAD DATA
# ============================================================

documents = load_documents()

all_chunks = build_knowledge_base(documents)

print(
    f"\nTotal chunks: {len(all_chunks)}"
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

texts = [
    chunk["text"]
    for chunk in all_chunks
]

print("Creating embeddings...")

embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True
)

print(
    "Embedding shape:",
    embeddings.shape
)


# ============================================================
# BM25
# ============================================================

print("\nBuilding BM25 index...")

tokenized_chunks = [
    tokenize(text)
    for text in texts
]

bm25 = BM25Okapi(
    tokenized_chunks
)


# ============================================================
# CROSS-ENCODER RERANKER
# ============================================================

print("\nLoading Cross-Encoder reranker...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

print("Reranker ready.")


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(scores):

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:

        return np.zeros_like(scores)

    return (
        (scores - minimum)
        /
        (maximum - minimum)
    )


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def hybrid_candidates(question):

    # Semantic scores
    query_embedding = embedding_model.encode(
        [question]
    )

    semantic_scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]


    # BM25 scores
    keyword_scores = np.array(
        bm25.get_scores(
            tokenize(question)
        )
    )


    # Normalize
    semantic_normalized = normalize(
        semantic_scores
    )

    keyword_normalized = normalize(
        keyword_scores
    )


    # Hybrid score
    hybrid_scores = (
        0.5 * semantic_normalized
        +
        0.5 * keyword_normalized
    )


    # Top candidates
    ranking = np.argsort(
        hybrid_scores
    )[::-1]

    return ranking[:CANDIDATE_K]


# ============================================================
# CROSS-ENCODER RERANKING
# ============================================================

def rerank(question, candidate_indices):

    pairs = []

    for index in candidate_indices:

        pairs.append(
            (
                question,
                all_chunks[index]["text"]
            )
        )


    scores = reranker.predict(
        pairs,
        show_progress_bar=False
    )


    # Sort highest score first
    sorted_positions = np.argsort(
        scores
    )[::-1]


    reranked_indices = [
        candidate_indices[position]
        for position in sorted_positions
    ]


    return reranked_indices


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def get_documents(ranking, k):

    selected = ranking[:k]

    return set(
        all_chunks[index]["document"]
        for index in selected
    )


# ============================================================
# RECALL
# ============================================================

def recall_at_k(
    ranking,
    gold_documents,
    k
):

    retrieved = get_documents(
        ranking,
        k
    )

    gold = set(
        gold_documents
    )

    found = retrieved & gold

    if not gold:
        return 0.0

    return len(found) / len(gold)


# ============================================================
# MRR
# ============================================================

def reciprocal_rank(
    ranking,
    gold_documents
):

    gold = set(
        gold_documents
    )

    for rank, index in enumerate(
        ranking,
        start=1
    ):

        document = all_chunks[
            index
        ]["document"]

        if document in gold:

            return 1 / rank

    return 0.0


# ============================================================
# RESULT STORAGE
# ============================================================

baseline_results = {

    "Recall@1": [],
    "Recall@3": [],
    "Recall@5": [],
    "MRR": []
}


reranked_results = {

    "Recall@1": [],
    "Recall@3": [],
    "Recall@5": [],
    "MRR": []
}


# ============================================================
# RUN 35 QUESTIONS
# ============================================================

print("\n")
print("=" * 70)
print(
    "HYBRID vs HYBRID + CROSS-ENCODER RERANKING"
)
print("=" * 70)


for number, (
    question_id,
    question,
    gold
) in enumerate(
    questions,
    start=1
):

    print(
        f"\n[{number}/35] {question_id}: "
        f"{question}"
    )


    # --------------------------------------------------------
    # Hybrid baseline
    # --------------------------------------------------------

    candidates = hybrid_candidates(
        question
    )


    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------

    reranked = rerank(
        question,
        candidates
    )


    # --------------------------------------------------------
    # Baseline metrics
    # --------------------------------------------------------

    for k in FINAL_K_VALUES:

        baseline_results[
            f"Recall@{k}"
        ].append(
            recall_at_k(
                candidates,
                gold,
                k
            )
        )


    baseline_results["MRR"].append(
        reciprocal_rank(
            candidates,
            gold
        )
    )


    # --------------------------------------------------------
    # Reranked metrics
    # --------------------------------------------------------

    for k in FINAL_K_VALUES:

        reranked_results[
            f"Recall@{k}"
        ].append(
            recall_at_k(
                reranked,
                gold,
                k
            )
        )


    reranked_results["MRR"].append(
        reciprocal_rank(
            reranked,
            gold
        )
    )


# ============================================================
# AVERAGES
# ============================================================

def average_results(results):

    return {

        metric: np.mean(values)

        for metric, values
        in results.items()
    }


baseline = average_results(
    baseline_results
)

reranked = average_results(
    reranked_results
)


# ============================================================
# FINAL TABLE
# ============================================================

print("\n\n")
print("=" * 80)
print("RERANKING EVALUATION RESULTS")
print("=" * 80)

print()

print(
    f"{'Method':<35}"
    f"{'Recall@1':>12}"
    f"{'Recall@3':>12}"
    f"{'Recall@5':>12}"
    f"{'MRR':>12}"
)

print("-" * 80)

print(
    f"{'Hybrid':<35}"
    f"{baseline['Recall@1']:>11.2%}"
    f"{baseline['Recall@3']:>11.2%}"
    f"{baseline['Recall@5']:>11.2%}"
    f"{baseline['MRR']:>11.3f}"
)

print(
    f"{'Hybrid + Reranker':<35}"
    f"{reranked['Recall@1']:>11.2%}"
    f"{reranked['Recall@3']:>11.2%}"
    f"{reranked['Recall@5']:>11.2%}"
    f"{reranked['MRR']:>11.3f}"
)


# ============================================================
# IMPROVEMENT
# ============================================================

print("\n")
print("=" * 80)
print("IMPROVEMENT FROM RERANKING")
print("=" * 80)


for metric in [
    "Recall@1",
    "Recall@3",
    "Recall@5",
    "MRR"
]:

    improvement = (
        reranked[metric]
        -
        baseline[metric]
    )

    if metric == "MRR":

        print(
            f"{metric:<12}: "
            f"{improvement:+.3f}"
        )

    else:

        print(
            f"{metric:<12}: "
            f"{improvement:+.2%}"
        )


# ============================================================
# RECOMMENDATION
# ============================================================

print("\n")
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)


if reranked["MRR"] > baseline["MRR"]:

    print(
        "\nReranking improved MRR."
    )

else:

    print(
        "\nReranking did not improve MRR."
    )


if reranked["Recall@3"] > baseline["Recall@3"]:

    print(
        "Reranking improved Recall@3."
    )

else:

    print(
        "Reranking did not improve Recall@3."
    )


if (
    reranked["MRR"] > baseline["MRR"]
    and
    reranked["Recall@3"] >= baseline["Recall@3"]
):

    print(
        "\nRecommended architecture:"
    )

    print(
        "Hybrid → Cross-Encoder Reranker"
    )

else:

    print(
        "\nRecommended architecture:"
    )

    print(
        "Hybrid retrieval without reranking"
    )


print("\nEvaluation completed.")