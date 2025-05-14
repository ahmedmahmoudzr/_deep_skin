import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# تحميل الموديل المحفوظ
model = load_model("skin_cancer_model.h5")

# المسار لمجلد الصور (مش لازم تستخدمه لو انت رافع الصور مباشرة)
srcdir = '/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_1'

# القاموس الخاص بالأصناف (هتكون موجودة بالفعل في مشروعك)
classes = {
    4: ('nv', ' melanocytic nevi'),
    6: ('mel', 'melanoma'),
    2: ('bkl', 'benign keratosis-like lesions'),
    1: ('bcc' , ' basal cell carcinoma'),
    5: ('vasc', ' pyogenic granulomas and hemorrhage'),
    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
    3: ('df', 'dermatofibroma')
}

# عرض الواجهة مع اختيار الصورة
st.title("Skin Cancer Detection")
image_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if image_file is not None:
    # قراءة الصورة من الملف
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)

    # تحويل BGR إلى RGB لعرضها بشكل صحيح
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # تصغير الصورة لـ 28x28 إذا الموديل مدرب على كده
    img_resized = cv2.resize(img, (28, 28))
    img_input = img_resized.reshape(1, 28, 28, 3) / 255.0

    # التنبؤ
    prediction = model.predict(img_input)
    class_index = np.argmax(prediction[0])
    class_name = classes[class_index][1]  # اسم التشخيص
    confidence = max(prediction[0]) * 100  # نسبة الثقة

    # عرض الصورة والنتيجة
    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
    st.write(f"Predicted Class: {class_name}")
    st.write(f"Confidence: {confidence:.2f}%")
