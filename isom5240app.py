from transformers import pipeline
from PIL import Image
import streamlit as st

def img_classify():
  age_classifier = pipeline("image-classification", model="ibombonato/swin-age-classifier")
  image_name = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
  image_name = Image.open(image_name).convert("RGB")
  # Classify age
  age_predictions = age_classifier(image_name)
  st.write(age_predictions)
  age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)

  return age_predictions


def main():
  st.header("Age classifier")

  img_classify()

  st.write("Predicted Age Range:")
  st.write(f"Age range: {age_predictions[0]['label']}")

if __name__ == "__main__":
    main()
