import pandas as pd

def preprocess_input(df):
    # تحويل التاريخ لو موجود
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        df['Year'] = df['Order Date'].dt.year
        df['Month'] = df['Order Date'].dt.month
        df['Weekday'] = df['Order Date'].dt.weekday
        df['IsWeekend'] = df['Weekday'].isin([5, 6]).astype(int)

    # التأكد من وجود PromotionFlag
    if 'PromotionFlag' not in df.columns:
        df['PromotionFlag'] = 0

    # تحميل أعمدة التدريب
    try:
        with open('training_columns.txt', 'r') as f:
            training_columns = f.read().split(',')
    except Exception as e:
        raise ValueError(f"فشل تحميل أعمدة التدريب: {e}")

    # إعادة ترتيب الأعمدة وملي الفراغات بـ 0
    final_data = df.reindex(columns=training_columns, fill_value=0)

    return final_data
