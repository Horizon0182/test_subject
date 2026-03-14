from transformers import pipeline
from PIL import Image
import streamlit as st

def img_classify(img_file):
  if img_file is None:
        st.warning("请先上传一张图片")
        return None
    
  age_classifier = pipeline("image-classification", model="ibombonato/swin-age-classifier")
  image = Image.open(img_file).convert("RGB")
    
  # 执行分类
  age_predictions = age_classifier(image)
    
  # 按置信度排序
  age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)
    
  return age_predictions


def main():
  st.header("Age classifier")
  img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
  age_predictions = img_classify(img)
    
  if age_predictions:
      st.write("Predicted Age Range:")
      st.write(f"Age range: {age_predictions[0]['label']}")
  else:
      st.write("无法预测年龄，请检查图片是否有效")

if __name__ == "__main__":
    main()



st.header("Testing app")
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import numpy as np

# Testing with the saved model
model2 = AutoModelForSequenceClassification.from_pretrained("Albatrosszzz/just_for_test",
                                                            num_labels=5)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenized testing data
label = 4 # label = 4
text = "dr. goldberg offers everything i look for in a general practitioner. he's nice and easy to talk to without being patronizing; he's always on time in seeing his patients; he's affiliated with a top-notch hospital (nyu) which my parents have explained to me is very important in case something happens and you need surgery; and you can get referrals to see specialists without having to see him first. really, what more do you need? i'm sitting here trying to think of any complaints i have about him, but i'm really drawing a blank."
inputs = tokenizer(text,
                   padding=True,
                   truncation=True,
                   return_tensors='pt')

outputs = model2(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
predictions = predictions.cpu().detach().numpy()

# Get the index of the largest output value
max_index = np.argmax(predictions)

st.write(f"The label is {label} and the predicted label is {max_index}")
