# 📦 Import thư viện
import streamlit as st
import os
import pdfplumber
import base64
import re
from transformers import pipeline
import pandas as pd

# ⚡ Set page config
st.set_page_config(page_title="Hệ thống Hỗ trợ Tuyển dụng bằng AI", layout="wide")

# 📥 Folder lưu CV
UPLOAD_FOLDER = './uploaded_cvs/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 📚 Load model AI
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_classifier()

FIELDS = ["IT"]

# --- Hàm xử lý ---
def extract_text_from_pdf(file_path):
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except:
        return ""

def predict_field(text_cv):
    short_text = text_cv[:1000]
    result = classifier(short_text, candidate_labels=FIELDS)
    return result['labels'][0]

def extract_email(text):
    match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    return match.group(0) if match else ""

def extract_name(text):
    lines = text.strip().split("\n")
    for line in lines:
        if len(line.split()) <= 5 and any(c.isalpha() for c in line):
            return line.strip()
    return "Không rõ"

def save_uploadedfile(uploadedfile):
    path = os.path.join(UPLOAD_FOLDER, uploadedfile.name)
    with open(path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return path

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    st.markdown(f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000"></iframe>', unsafe_allow_html=True)

# --- Giao diện chính ---
def main():
    st.title("📄 Hệ thống Hỗ trợ Quản lý Tuyển dụng bằng AI")

    st.sidebar.header("📤 Upload CV (.pdf)")
    uploaded_files = st.sidebar.file_uploader("Tải lên nhiều CV", type=["pdf"], accept_multiple_files=True)

    cv_data = []

    if uploaded_files:
        uploaded_paths = []
        for uploaded_file in uploaded_files:
            path = save_uploadedfile(uploaded_file)
            uploaded_paths.append(path)
        st.sidebar.success(f"✅ Đã upload {len(uploaded_files)} file CV.")

        st.success("✅ Đang tiến hành phân tích CV...")
        result_placeholder = st.empty()
        my_bar = st.progress(0)

        for idx, file_path in enumerate(uploaded_paths):
            text = extract_text_from_pdf(file_path)

            if text:
                try:
                    predicted_field = predict_field(text)
                    name = extract_name(text)
                    email = extract_email(text)

                    cv_data.append({
                        'Tên file': os.path.basename(file_path),
                        'Tên ứng viên': name,
                        'Email': email,
                        'Nội dung CV': text,
                        'Ngành nghề': predicted_field,
                        'Đường dẫn': file_path
                    })
                except Exception as e:
                    st.error(f"Lỗi phân tích file {file_path}: {str(e)}")

            my_bar.progress((idx + 1) / len(uploaded_paths))
            result_placeholder.dataframe(pd.DataFrame(cv_data)[['Tên file', 'Tên ứng viên', 'Email', 'Ngành nghề']])

        my_bar.empty()
        st.toast("✅ Đã hoàn tất phân tích CV!", icon="✅")

    if cv_data:
        df = pd.DataFrame(cv_data)

        st.subheader("🌟 Chọn ngành để lọc")
        selected_field = st.selectbox("Ngành nghề", FIELDS)

        filtered_df = df[df['Ngành nghề'] == selected_field]

        st.subheader(f"📋 Danh sách ứng viên ngành {selected_field}")
        st.dataframe(filtered_df[['Tên file', 'Tên ứng viên', 'Email', 'Ngành nghề']])

        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Tải danh sách ứng viên",
            data=csv,
            file_name=f'cv_filtered_{selected_field}.csv',
            mime='text/csv',
        )

        st.markdown("---")

        st.subheader("🔍 Phân tích chi tiết CV")
        selected_cv_file = st.selectbox("📂 Chọn CV để phân tích", filtered_df['Tên file'])

        if selected_cv_file:
            selected_cv = filtered_df[filtered_df['Tên file'] == selected_cv_file].iloc[0]
            st.info(f"🔸 Đang phân tích CV: {selected_cv['Tên file']}")

            st.subheader("🖼️ Xem nội dung CV")
            show_pdf(selected_cv['Đường dẫn'])

            st.subheader("🧐 Đánh giá kỹ năng ngành IT")
            skills = ["python", "java", "c++", "javascript", "nodejs", "react", "sql", "django"]
            skill_count = sum(skill.lower() in selected_cv['Nội dung CV'].lower() for skill in skills)

            st.success(f"🌟 Kỹ năng phù hợp: {skill_count}/{len(skills)}")

            if skill_count >= 5:
                st.success("✅ Ứng viên rất phù hợp với ngành IT.")
            elif skill_count >= 2:
                st.info("➖ Ứng viên có một số kỹ năng phù hợp.")
            else:
                st.warning("⚠️ Ứng viên thiếu kỹ năng cần thiết.")

if __name__ == "__main__":
    main()
