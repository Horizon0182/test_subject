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
