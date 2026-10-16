import csv
import time
import main


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


output_file = "answer_evaluation_35.csv"


print("=" * 70)
print("FOOD SAFETY RAG - ANSWER EVALUATION")
print("=" * 70)

results = []


for number, question in enumerate(questions, start=1):

    print("\n" + "-" * 70)
    print(f"Question {number}/35")
    print(question)

    try:

        # Retrieve context using the existing RAG system
        search_query = main.build_search_query(
            question,
            None
        )

        selected_chunks, reranked, best_score = (
            main.retrieve_context(search_query)
        )

        # If retrieval threshold rejects the question
        if selected_chunks is None:

            answer = (
                "I couldn't find sufficient information to "
                "answer this question in the available "
                "food-safety documents."
            )

            error_type = "retrieval_threshold"
            sources = ""

        else:

            # Generate grounded Gemini answer
            answer, error_type = main.generate_answer(
                question,
                selected_chunks
            )

            # Collect source documents
            source_list = []

            for index in selected_chunks:

                filename = main.all_chunks[index]["document"]

                source = main.clean_source_name(filename)

                if source not in source_list:
                    source_list.append(source)

            sources = "; ".join(source_list)

        print("\nANSWER:")
        print(answer)

        print("\nSOURCES:")
        print(sources)

        print("\nBEST RETRIEVAL SCORE:")
        print(best_score)

        results.append({
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

        results.append({
            "question_id": number,
            "question": question,
            "answer": "",
            "sources": "",
            "best_retrieval_score": "",
            "error_type": str(e)
        })

    # Wait before the next Gemini request
    # This helps avoid Gemini rate limits.
    if number < len(questions):

        print("\nWaiting 15 seconds before next question...")

        time.sleep(15)


# Save all answers
with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "question_id",
            "question",
            "answer",
            "sources",
            "best_retrieval_score",
            "error_type"
        ]
    )

    writer.writeheader()

    writer.writerows(results)


print("\n" + "=" * 70)
print("ANSWER EVALUATION COMPLETED")
print("=" * 70)

print(f"\nSaved file: {output_file}")

print("\nThe CSV contains:")
print("1. Question")
print("2. Gemini Answer")
print("3. Source Documents")
print("4. Retrieval Score")
print("5. Error Type")

print("\nNext stage:")
print("Evaluate correctness, relevance, context support,")
print("hallucination, and source accuracy.")