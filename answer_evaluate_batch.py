import csv
import time
import main


# ---------------------------------------------------------
# 35 QUESTIONS
# ---------------------------------------------------------

questions = [
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
    "Can a food product be considered safe if it exceeds a contaminant limit?"
]


# ---------------------------------------------------------
# FILE
# ---------------------------------------------------------

output_file = "answer_evaluation_35.csv"


# ---------------------------------------------------------
# BATCH SETTINGS
# ---------------------------------------------------------

BATCH_SIZE = 5

# Wait between Gemini questions
WAIT_SECONDS = 20


# ---------------------------------------------------------
# LOAD EXISTING CSV
# ---------------------------------------------------------

try:

    with open(
        output_file,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)
        existing_rows = list(reader)

except FileNotFoundError:

    existing_rows = []


# ---------------------------------------------------------
# CREATE EXISTING ANSWER MAP
# ---------------------------------------------------------

existing_answers = {}

for row in existing_rows:

    try:
        question_id = int(row["question_id"])
    except:
        continue

    answer = row.get("answer", "").strip()

    error_type = row.get("error_type", "").strip()

    # Consider an answer successful only if it has
    # actual generated content and no API error.
    if answer and error_type not in [
        "rate_limit",
        "api_error",
        "unavailable"
    ]:

        existing_answers[question_id] = row


# ---------------------------------------------------------
# FIND QUESTIONS STILL NEEDING ANSWERS
# ---------------------------------------------------------

remaining = []

for number, question in enumerate(questions, start=1):

    if number not in existing_answers:
        remaining.append((number, question))


print("=" * 70)
print("FOOD SAFETY RAG - BATCH ANSWER EVALUATION")
print("=" * 70)

print(f"\nTotal questions: {len(questions)}")
print(f"Already completed: {len(existing_answers)}")
print(f"Remaining: {len(remaining)}")


if not remaining:

    print("\nAll 35 questions already have answers.")
    print("Nothing more to generate.")
    exit()


# ---------------------------------------------------------
# PROCESS ONLY ONE BATCH
# ---------------------------------------------------------

batch = remaining[:BATCH_SIZE]

print(f"\nThis run will process {len(batch)} questions:")
print(
    ", ".join(
        f"Q{number}"
        for number, question in batch
    )
)


new_rows = []


for position, (number, question) in enumerate(batch, start=1):

    print("\n" + "-" * 70)

    print(
        f"Processing Q{number} "
        f"({position}/{len(batch)})"
    )

    print(question)

    try:

        # ---------------------------------------------
        # RETRIEVAL
        # ---------------------------------------------

        search_query = main.build_search_query(
            question,
            None
        )

        (
            selected_chunks,
            reranked,
            best_score
        ) = main.retrieve_context(search_query)


        # ---------------------------------------------
        # RETRIEVAL FAILED
        # ---------------------------------------------

        if selected_chunks is None:

            answer = (
                "I couldn't find sufficient information to "
                "answer this question in the available "
                "food-safety documents."
            )

            error_type = "retrieval_threshold"

            sources = ""


        # ---------------------------------------------
        # GEMINI
        # ---------------------------------------------

        else:

            answer, error_type = main.generate_answer(
                question,
                selected_chunks
            )

            source_list = []

            for index in selected_chunks:

                filename = (
                    main.all_chunks[index]["document"]
                )

                source = main.clean_source_name(
                    filename
                )

                if source not in source_list:
                    source_list.append(source)

            sources = "; ".join(source_list)


        # ---------------------------------------------
        # PRINT
        # ---------------------------------------------

        print("\nANSWER:")
        print(answer)

        print("\nSOURCES:")
        print(sources)

        print("\nRETRIEVAL SCORE:")
        print(best_score)

        print("\nERROR TYPE:")
        print(error_type)


        # ---------------------------------------------
        # SAVE RESULT
        # ---------------------------------------------

        new_rows.append({
            "question_id": number,
            "question": question,
            "answer": answer,
            "sources": sources,
            "best_retrieval_score": best_score,
            "error_type": error_type
        })


    except Exception as e:

        print("\nERROR:")
        print(e)

        new_rows.append({
            "question_id": number,
            "question": question,
            "answer": "",
            "sources": "",
            "best_retrieval_score": "",
            "error_type": str(e)
        })


    # ---------------------------------------------
    # WAIT
    # ---------------------------------------------

    if position < len(batch):

        print(
            f"\nWaiting {WAIT_SECONDS} seconds..."
        )

        time.sleep(WAIT_SECONDS)


# ---------------------------------------------------------
# MERGE OLD + NEW RESULTS
# ---------------------------------------------------------

all_rows = {}

for row in existing_rows:

    try:
        question_id = int(row["question_id"])
        all_rows[question_id] = row
    except:
        pass


for row in new_rows:

    question_id = int(row["question_id"])

    all_rows[question_id] = row


# ---------------------------------------------------------
# SAVE CSV
# ---------------------------------------------------------

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    fieldnames = [
        "question_id",
        "question",
        "answer",
        "sources",
        "best_retrieval_score",
        "error_type"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for question_id in sorted(all_rows):

        writer.writerow(
            all_rows[question_id]
        )


# ---------------------------------------------------------
# STATUS
# ---------------------------------------------------------

completed = 0

for question_id in range(1, len(questions) + 1):

    if question_id in all_rows:

        answer = all_rows[question_id].get(
            "answer",
            ""
        ).strip()

        error = all_rows[question_id].get(
            "error_type",
            ""
        ).strip()

        if (
            answer
            and error not in [
                "rate_limit",
                "api_error",
                "unavailable"
            ]
        ):

            completed += 1


remaining_after = len(questions) - completed


print("\n" + "=" * 70)
print("BATCH COMPLETED")
print("=" * 70)

print(f"\nCompleted answers: {completed}/35")
print(f"Remaining answers: {remaining_after}")

print(f"\nUpdated file:")
print(output_file)

if remaining_after > 0:

    print(
        "\nRun this same command again for the next batch:"
    )

    print(
        "\npython answer_evaluate_batch.py"
    )

else:

    print(
        "\nALL 35 ANSWERS ARE READY FOR FINAL EVALUATION!"
    )