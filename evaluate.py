import os
import re
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
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

RRF_K = 60

TOP_K_VALUES = [1, 3, 5]


# ============================================================
# 35 EVALUATION QUESTIONS
# ============================================================

questions = [

    # --------------------------------------------------------
    # E1 - Direct Questions
    # --------------------------------------------------------

    {
        "id": "Q01",
        "question": "What is food hygiene according to Codex?",
        "gold": ["cc6125en.pdf"]
    },

    {
        "id": "Q02",
        "question": "What are Good Hygiene Practices (GHPs)?",
        "gold": ["cc6125en.pdf"]
    },

    {
        "id": "Q03",
        "question": "Why is training important in a food hygiene system?",
        "gold": ["cc6125en.pdf"]
    },

    {
        "id": "Q04",
        "question": "What should cleaning accomplish?",
        "gold": [
            "cc6125en.pdf",
            "Licensing_Regulations.pdf"
        ]
    },

    {
        "id": "Q05",
        "question": "What is the requirement for petty food business registration?",
        "gold": ["Licensing_Regulations.pdf"]
    },

    {
        "id": "Q06",
        "question": "What are the basic requirements for a petty food manufacturer?",
        "gold": ["Licensing_Regulations.pdf"]
    },

    {
        "id": "Q07",
        "question": "What are the limits for aflatoxin and aflatoxin M1?",
        "gold": ["Contaminants_Regulations.pdf"]
    },

    {
        "id": "Q08",
        "question": "What is the mercury limit in fish?",
        "gold": ["Contaminants_Regulations.pdf"]
    },

    {
        "id": "Q09",
        "question": "What is meant by frozen dessert or frozen confection?",
        "gold": ["Food_Additives_Regulations.pdf"]
    },

    {
        "id": "Q10",
        "question": "What are the microbiological requirements for food products?",
        "gold": [
            "Food_Additives_Regulations.pdf",
            "cc6125en.pdf"
        ]
    },


    # --------------------------------------------------------
    # E2 - Paraphrased Questions
    # --------------------------------------------------------

    {
        "id": "Q11",
        "question": "Why should food workers receive hygiene training?",
        "gold": ["cc6125en.pdf"]
    },

    {
        "id": "Q12",
        "question": "How does cleaning food-contact surfaces prevent contamination?",
        "gold": [
            "cc6125en.pdf",
            "Licensing_Regulations.pdf"
        ]
    },

    {
        "id": "Q13",
        "question": "How can contamination between raw and ready-to-eat food be reduced?",
        "gold": [
            "cc6125en.pdf",
            "Licensing_Regulations.pdf"
        ]
    },

    {
        "id": "Q14",
        "question": "What must a small food business do before operating?",
        "gold": ["Licensing_Regulations.pdf"]
    },

    {
        "id": "Q15",
        "question": "What happens if contaminants exceed the specified FSSAI limits?",
        "gold": ["Contaminants_Regulations.pdf"]
    },

    {
        "id": "Q16",
        "question": "What are naturally occurring toxic substances and their limits?",
        "gold": ["Contaminants_Regulations.pdf"]
    },

    {
        "id": "Q17",
        "question": "What food products are covered by frozen dessert standards?",
        "gold": ["Food_Additives_Regulations.pdf"]
    },

    {
        "id": "Q18",
        "question": "Why are food additives mentioned in the regulations?",
        "gold": ["Food_Additives_Regulations.pdf"]
    },


    # --------------------------------------------------------
    # E3 - Keyword / Regulatory Questions
    # --------------------------------------------------------

    {
        "id": "Q19",
        "question": "What is a prerequisite programme?",
        "gold": ["cc6125en.pdf"]
    },

    {
        "id": "Q20",
        "question": "What is a significant hazard?",
        "gold": ["cc6125en.pdf"]
    },

    {
        "id": "Q21",
        "question": "What is HACCP?",
        "gold": ["cc6125en.pdf"]
    },

    {
        "id": "Q22",
        "question": "What is a Registering Authority?",
        "gold": ["Licensing_Regulations.pdf"]
    },

    {
        "id": "Q23",
        "question": "What is a State Licensing Authority?",
        "gold": ["Licensing_Regulations.pdf"]
    },

    {
        "id": "Q24",
        "question": "What is aflatoxin M1 and what is its limit in milk?",
        "gold": ["Contaminants_Regulations.pdf"]
    },

    {
        "id": "Q25",
        "question": "What is the cadmium limit in turmeric?",
        "gold": ["Contaminants_Regulations.pdf"]
    },

    {
        "id": "Q26",
        "question": "What is methyl mercury?",
        "gold": ["Contaminants_Regulations.pdf"]
    },


    # --------------------------------------------------------
    # E4 - Multi-document Questions
    # --------------------------------------------------------

    {
        "id": "Q27",
        "question": "What are Codex hygiene requirements and FSSAI food-business requirements?",
        "gold": [
            "cc6125en.pdf",
            "Licensing_Regulations.pdf"
        ]
    },

    {
        "id": "Q28",
        "question": "How are contamination prevention requirements different from contaminant limits?",
        "gold": [
            "cc6125en.pdf",
            "Contaminants_Regulations.pdf"
        ]
    },

    {
        "id": "Q29",
        "question": "What are the hygiene responsibilities of a food business and how do contaminant limits relate to them?",
        "gold": [
            "cc6125en.pdf",
            "Licensing_Regulations.pdf",
            "Contaminants_Regulations.pdf"
        ]
    },

    {
        "id": "Q30",
        "question": "How do hygiene practices relate to food product standards?",
        "gold": [
            "cc6125en.pdf",
            "Food_Additives_Regulations.pdf"
        ]
    },


    # --------------------------------------------------------
    # E5 - Hard Retrieval Questions
    # --------------------------------------------------------

    {
        "id": "Q31",
        "question": "Why is food hygiene training necessary?",
        "gold": ["cc6125en.pdf"]
    },

    {
        "id": "Q32",
        "question": "How should raw food and ready-to-eat food be separated?",
        "gold": [
            "cc6125en.pdf",
            "Licensing_Regulations.pdf"
        ]
    },

    {
        "id": "Q33",
        "question": "What is the difference between cleaning and disinfection?",
        "gold": [
            "cc6125en.pdf",
            "Licensing_Regulations.pdf"
        ]
    },

    {
        "id": "Q34",
        "question": "What are the registration and hygiene obligations of a small food business?",
        "gold": ["Licensing_Regulations.pdf"]
    },

    {
        "id": "Q35",
        "question": "Can a food product be considered safe if it exceeds a contaminant limit?",
        "gold": ["Contaminants_Regulations.pdf"]
    }
]


# ============================================================
# TOKENIZATION
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

        filepath = os.path.join(
            DOCUMENT_FOLDER,
            filename
        )

        if not os.path.exists(filepath):

            print("Missing:", filepath)

            continue

        reader = PdfReader(filepath)

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

    all_chunks = []

    for filename, text in documents.items():

        chunks = create_chunks(text)

        for i, chunk in enumerate(chunks):

            all_chunks.append({
                "document": filename,
                "chunk_id": i,
                "text": chunk
            })

    return all_chunks


# ============================================================
# LOAD
# ============================================================

documents = load_documents()

all_chunks = build_knowledge_base(documents)

print(
    f"\nTotal chunks: {len(all_chunks)}"
)


# ============================================================
# EMBEDDINGS
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

texts = [
    chunk["text"]
    for chunk in all_chunks
]

print("Creating embeddings...")

embeddings = model.encode(
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
    tokenize(chunk["text"])
    for chunk in all_chunks
]

bm25 = BM25Okapi(
    tokenized_chunks
)


# ============================================================
# RRF
# ============================================================

def reciprocal_rank_fusion(
    rankings,
    k=RRF_K
):

    scores = np.zeros(
        len(all_chunks)
    )

    for ranking in rankings:

        for rank, index in enumerate(
            ranking,
            start=1
        ):

            scores[index] += (
                1 / (k + rank)
            )

    return scores


# ============================================================
# SEARCH METHODS
# ============================================================

def semantic_search(question):

    query_embedding = model.encode(
        [question]
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    ranking = np.argsort(
        scores
    )[::-1]

    return ranking


def bm25_search(question):

    scores = np.array(
        bm25.get_scores(
            tokenize(question)
        )
    )

    ranking = np.argsort(
        scores
    )[::-1]

    return ranking


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


def hybrid_search(question):

    query_embedding = model.encode(
        [question]
    )

    semantic_scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    keyword_scores = np.array(
        bm25.get_scores(
            tokenize(question)
        )
    )

    semantic_normalized = normalize(
        semantic_scores
    )

    keyword_normalized = normalize(
        keyword_scores
    )

    hybrid_scores = (
        0.5 * semantic_normalized
        +
        0.5 * keyword_normalized
    )

    ranking = np.argsort(
        hybrid_scores
    )[::-1]

    return ranking


def rrf_search(question):

    semantic_ranking = semantic_search(
        question
    )

    bm25_ranking = bm25_search(
        question
    )

    rrf_scores = reciprocal_rank_fusion(
        [
            semantic_ranking,
            bm25_ranking
        ],
        RRF_K
    )

    ranking = np.argsort(
        rrf_scores
    )[::-1]

    return ranking


# ============================================================
# DOCUMENT-LEVEL EVALUATION
# ============================================================

def get_documents(ranking, k):

    selected = ranking[:k]

    return set(
        all_chunks[index]["document"]
        for index in selected
    )


def recall_at_k(
    ranking,
    gold_documents,
    k
):

    retrieved_documents = get_documents(
        ranking,
        k
    )

    found = (
        retrieved_documents
        &
        set(gold_documents)
    )

    if len(gold_documents) == 0:

        return 0.0

    return (
        len(found)
        /
        len(gold_documents)
    )


def reciprocal_rank(
    ranking,
    gold_documents
):

    gold_documents = set(
        gold_documents
    )

    for rank, index in enumerate(
        ranking,
        start=1
    ):

        document = all_chunks[
            index
        ]["document"]

        if document in gold_documents:

            return 1 / rank

    return 0.0


# ============================================================
# METHODS
# ============================================================

methods = {

    "Semantic Search": semantic_search,

    "BM25": bm25_search,

    "Hybrid": hybrid_search,

    "RRF": rrf_search
}


# ============================================================
# RESULTS STORAGE
# ============================================================

results = {}


for method_name in methods:

    results[method_name] = {

        "Recall@1": [],

        "Recall@3": [],

        "Recall@5": [],

        "MRR": []
    }


# ============================================================
# RUN EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("RUNNING 35-QUESTION RETRIEVAL EVALUATION")
print("=" * 70)


for number, item in enumerate(
    questions,
    start=1
):

    question_id = item["id"]

    question = item["question"]

    gold = item["gold"]

    print(
        f"\n[{number}/35] {question_id}: "
        f"{question}"
    )

    for method_name, search_function in methods.items():

        ranking = search_function(
            question
        )

        r1 = recall_at_k(
            ranking,
            gold,
            1
        )

        r3 = recall_at_k(
            ranking,
            gold,
            3
        )

        r5 = recall_at_k(
            ranking,
            gold,
            5
        )

        mrr = reciprocal_rank(
            ranking,
            gold
        )

        results[method_name][
            "Recall@1"
        ].append(r1)

        results[method_name][
            "Recall@3"
        ].append(r3)

        results[method_name][
            "Recall@5"
        ].append(r5)

        results[method_name][
            "MRR"
        ].append(mrr)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n\n")
print("=" * 80)
print("FINAL RETRIEVAL EVALUATION")
print("=" * 80)

print()

header = (
    f"{'Method':<22}"
    f"{'Recall@1':>12}"
    f"{'Recall@3':>12}"
    f"{'Recall@5':>12}"
    f"{'MRR':>12}"
)

print(header)

print("-" * 80)


method_averages = {}


for method_name, metrics in results.items():

    recall1 = np.mean(
        metrics["Recall@1"]
    )

    recall3 = np.mean(
        metrics["Recall@3"]
    )

    recall5 = np.mean(
        metrics["Recall@5"]
    )

    mrr = np.mean(
        metrics["MRR"]
    )

    method_averages[
        method_name
    ] = {

        "Recall@1": recall1,

        "Recall@3": recall3,

        "Recall@5": recall5,

        "MRR": mrr
    }

    print(
        f"{method_name:<22}"
        f"{recall1:>11.2%}"
        f"{recall3:>11.2%}"
        f"{recall5:>11.2%}"
        f"{mrr:>11.3f}"
    )


# ============================================================
# WINNERS
# ============================================================

print("\n")
print("=" * 80)
print("BEST METHOD BY METRIC")
print("=" * 80)


for metric in [
    "Recall@1",
    "Recall@3",
    "Recall@5",
    "MRR"
]:

    winner = max(
        method_averages,
        key=lambda method:
            method_averages[
                method
            ][metric]
    )

    score = method_averages[
        winner
    ][metric]

    print(
        f"{metric:<12}: "
        f"{winner:<22} "
        f"{score:.2%}"
        if metric != "MRR"
        else
        f"{metric:<12}: "
        f"{winner:<22} "
        f"{score:.3f}"
    )


# ============================================================
# RECOMMENDATION
# ============================================================

print("\n")
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)

best_overall = max(
    method_averages,
    key=lambda method:
        (
            method_averages[method]["Recall@3"]
            +
            method_averages[method]["MRR"]
        )
)

print(
    f"\nRecommended retrieval method: "
    f"{best_overall}"
)

print(
    "\nThis evaluation is document-level."
)

print(
    "It measures whether the retrieved documents "
    "contain the expected source documents."
)

print(
    "\nFor the final project, use these results "
    "together with your earlier reranking and "
    "RRF-k experiments to justify the architecture."
)

print("\nEvaluation completed.")