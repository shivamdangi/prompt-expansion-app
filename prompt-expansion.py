import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextGenerationPipeline

# Function to load tokenizer and model
def load_model():
    tokenizer_path = r'path_to_save'
    model_path = r'path_to_save'

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    return tokenizer, model

# Function to generate extended prompt
def generate_extended_prompt(prompt, tokenizer, model):
    output_prompt = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer
    )
    extended_prompt = output_prompt(prompt + ',', num_return_sequences=1)[0]["generated_text"]
    return extended_prompt

def main():
    st.title("Prompt Enhancement Model App")

    # Load tokenizer and model
    tokenizer, model = load_model()

    # Input prompt from user
    prompt = st.text_area("Enter your prompt:")

    if st.button("Generate Extended Prompt"):
        if prompt:
            extended_prompt = generate_extended_prompt(prompt, tokenizer, model)
            st.success("Extended Prompt:")
            st.write(extended_prompt)
        else:
            st.warning("Please enter a prompt.")

if __name__ == "__main__":
    main()
