import streamlit as st
import pandas as pd
import joblib

# تحميل النموذج والمحولات
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
onehot_encoders = joblib.load("onehot_encoders.pkl")

# تحميل الأعمدة المطلوبة كما كانت وقت التدريب
with open("training_columns.txt", "r") as f:
    training_columns = f.read().split(",")

# العنوان
st.title("Sales Forecasting System")
st.subheader("Please enter the following data:")


# مدخلات المستخدم
order_date_str = st.text_input("Order date(Order Date)", value="2023-01-01")  # يمكن للمستخدم إدخال التاريخ بأي صيغة
segment = st.selectbox(" (Segment)", ["Consumer", "Corporate", "Home Office"])
promotion_flag_str = st.selectbox("Is there a promotion?(PromotionFlag)", ["Yes", "No"])
promotion_flag = 1 if promotion_flag_str == "Yes" else 0
product_id = st.text_input(" (Product ID)")
country = st.text_input(" (Country)")

if st.button("Sales forecast"):
    try:
        # تحويل النص المدخل إلى تاريخ
        order_date = pd.to_datetime(order_date_str, errors='coerce')

        if pd.isna(order_date):
            st.error("Invalid date! Please enter a valid date.")
        else:
            # إنشاء DataFrame مبدأي
            input_data = pd.DataFrame([{
                "Order Date": order_date,
                "Segment": segment,
                "PromotionFlag": int(promotion_flag),
                "Product ID": product_id,
                "Country": country
            }])

            # معالجة التاريخ
            input_data["Year"] = input_data["Order Date"].dt.year
            input_data["Month"] = input_data["Order Date"].dt.month
            input_data["Weekday"] = input_data["Order Date"].dt.weekday
            input_data["IsWeekend"] = input_data["Weekday"].isin([5, 6])

            # ترميز الأعمدة
            input_data["Country"] = encoders["Country"].transform(input_data["Country"])
            input_data["Product ID"] = encoders["Product"].transform(input_data["Product ID"])

            # One-Hot Encoding للقطاع
            segment_encoded = onehot_encoders["Segment"].transform(input_data[["Segment"]]).toarray()
            segment_df = pd.DataFrame(
                segment_encoded,
                columns=onehot_encoders["Segment"].get_feature_names_out(["Segment"]),
                index=input_data.index
            )

            # دمج الأعمدة المعالجة
            processed = pd.concat([
                input_data[["Year", "Month", "Weekday", "IsWeekend", "PromotionFlag", "Country", "Product ID"]],
                segment_df
            ], axis=1)

            # إعادة ترتيب الأعمدة وإضافة الأعمدة الناقصة كـ 0
            processed = processed.reindex(columns=training_columns, fill_value=0)

            # التنبؤ
            prediction = model.predict(processed)
            st.success(f"القيمة المتوقعة للمبيعات: {prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"حدث خطأ أثناء التنبؤ: {e}")
