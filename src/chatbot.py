import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging

# Suppress noise in terminal
logging.set_verbosity_error()

def run_chatbot():
    """
    Stateful AI Wrapper using Blenderbot-400M.
    Implements a simple sliding window for context persistence.
    """
    model_id = "facebook/blenderbot-400M-distill"
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    history = []
    print(f"[*] Chatbot ready. Type 'exit' to quit.")

    while True:
        user_input = input("\nYou > ").strip()
        if not user_input or user_input.lower() in ["exit", "quit", "q"]:
            break

        # Context: Blenderbot uses double spaces as turn separators
        context = "  ".join(history)
        full_input = f"{context}  {user_input}" if history else user_input

        # Inference pipeline
        inputs = tokenizer(full_input, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.3, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        print(f"Bot > {response}")

        # Update and prune history (last 3 turns only)
        history.extend([user_input, response])
        history = history[-6:]

if __name__ == "__main__":
    run_chatbot()