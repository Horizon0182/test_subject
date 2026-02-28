from transformers import pipeline
from PIL import Image
import streamlit as st

def img_classify(img_file):
  age_classifier = pipeline("image-classification", model="ibombonato/swin-age-classifier")
  img_file = img_file.open(img_file).convert("RGB")
  
  # Classify age
  age_predictions = age_classifier(img_file)

  st.write(age_predictions)
  
  age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)

  return age_predictions


def main():
  st.header("Age classifier")
  img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

  img_classify(img_file)

  st.write("Predicted Age Range:")
  st.write(f"Age range: {age_predictions[0]['label']}")

if __name__ == "__main__":
    main()
