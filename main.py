from rag.rag_utils import ask_doctor_chat, add_papers_to_chroma
from utils.api_utils import fetch_semantic_scholar

def main():
    def doc_chat():
        print("Welcome to the Doctor Chatbot! Type 'exit' to quit.")

        while True:
            # Ask for the doctor's query
            
            query = input("\nEnter your question: ")
            if query == "":
                print("Goodbye!")
                break

            # Call your smart RAG function
            answer = ask_doctor_chat(query, top_k=5)
            print("\n--- Answer ---")
            print(answer)
    doc_chat()

if __name__ == "__main__":
    main()

