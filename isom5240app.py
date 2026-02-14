import streamlit as st

st.write("ISOM5240")
st.write("ISOM5240")

from transformers import pipeline

# Load text generation pipeline
# Specify the model you want to use
generator = pipeline("text-generation",
                     model="distilbert/distilgpt2")

# Generate text
prompt = "Once upon a time in a land far, far away"
generated_story = generator(prompt,
                            max_length=150)

# Output
print("Generated Story:")
print(generated_story[0]['generated_text'])
