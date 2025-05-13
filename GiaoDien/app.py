# 📦 Import thư viện
import streamlit as st
import os
import pdfplumber
import re
import base64
from transformers import pipeline
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# ⚡ Cấu hình trang
st.set_page_config(page_title="Hệ thống Hỗ trợ Tuyển dụng bằng AI", layout="wide")

# 📅 Thư mục lưu CV
UPLOAD_FOLDER = './uploaded_cvs/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 📚 Load mô hình AI
@st.cache_resource
def load_classifier():
    try:
        return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình AI: {str(e)}. Vui lòng kiểm tra kế nối mạng hoặc thử lại sau.")
        return None

classifier = load_classifier()
FIELDS = ["Frontend Development", "Backend Development", "Data Science/AI", "DevOps", "Mobile Development"]

# --- Hàm lưu file ---
def save_uploadedfile(uploadedfile):
    path = os.path.join(UPLOAD_FOLDER, uploadedfile.name)
    with open(path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return path

# --- Hàm xử lý PDF ---
def extract_text_from_pdf(file_path):
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages[:3]:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Lỗi khi đọc file PDF: {str(e)}")
        return ""

# --- Trích xuất tên ---
def extract_name(text):
    lines = text.strip().split("\n")
    for line in lines[:10]:
        if re.search(r"(Name|Tên):", line, re.IGNORECASE):
            return line.split(":")[-1].strip()
    for line in lines[:10]:
        if len(line.split()) >= 2 and line[0].isupper():
            if not any(char.isdigit() for char in line) and len(line.split()) <= 5 and not any(kw in line.lower() for kw in ["contact", "information"]):
                return line.strip()
    return "Không rõ"

# --- Phân loại lĩnh vực ---
def predict_field(text_cv):
    if classifier is None:
        return "Không xác định"
    short_text = text_cv[:300]
    result = classifier(short_text, candidate_labels=FIELDS)
    return result['labels'][0]

# --- Trích xuất kỹ năng từ CV (từ mục kỹ năng) ---
def extract_skills_list(text):
    skills = []
    for line in text.splitlines():
        if re.search(r'(skill|tools|tech|technology|framework)', line, re.IGNORECASE):
            parts = re.split(r'[:,]', line)
            if len(parts) > 1:
                items = re.split(r'[,/]', parts[1])
                items = [item.strip(" ()").strip() for item in items if item.strip()]
                skills.extend(items)
    return list(set(skills))

# --- Trích xuất kỹ năng sử dụng trong project ---
def extract_skills_from_projects(text):
    sections = re.findall(r"(?i)(project|dự án)[^\n]*\n+(.*?)(?=\n{2,}|\Z)", text, re.DOTALL)
    all_skills = set()

    for _, section in sections:
        lines = section.split('\n')
        for line in lines:
            if any(kw in line.lower() for kw in ['stack', 'tech', 'technology', 'tools', 'framework', 'sử dụng']):
                items = re.split(r'[:,]', line)
                if len(items) > 1:
                    for part in re.split(r'[,/•]', items[1]):
                        skill = part.strip(" -•()")
                        if 1 < len(skill) <= 30:
                            all_skills.add(skill)
    return sorted(all_skills)

# --- So khớp kỹ năng ---
def match_skills_accurately(candidate_skills, expected_skills):
    matched = [s for s in expected_skills if any(s.lower() in c.lower() for c in candidate_skills)]
    missing = [s for s in expected_skills if s not in matched]
    coverage = round(len(matched) / len(expected_skills) * 100, 2) if expected_skills else 0
    return matched, missing, coverage

# --- So khớp nghề ---
def match_field(text_cv, target_field):
    predicted_field = predict_field(text_cv)
    return predicted_field.lower() == target_field.lower()

# --- Hiển thị PDF ---
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# --- Phân tích một CV ---
def process_cv(file_path, expected_skills, target_field):
    text = extract_text_from_pdf(file_path)
    if text:
        if not match_field(text, target_field):
            return None

        name = extract_name(text)

        candidate_skills = extract_skills_list(text)
        project_skills = extract_skills_from_projects(text)
        total_skills = list(set(candidate_skills + project_skills))

        matched, missing, skill_coverage = match_skills_accurately(total_skills, expected_skills)

        final_coverage = skill_coverage
        if final_coverage == 0:
            return None

        result_status = "Phù hợp" if final_coverage >= 50 else "Không phù hợp"

        return {
            'Tên file': os.path.basename(file_path),
            'Tên ứng viên': name,
            'Mảng IT': target_field,
            'Phần trăm phù hợp': final_coverage,
            'Kết quả': result_status,
            'Kỹ năng phù hợp': ', '.join(matched),
            'Kỹ năng còn thiếu': ', '.join(missing),
            'Kỹ năng trong project': ', '.join(project_skills)
        }
    return None

# --- Phân tích nhiều CV ---
@st.cache_data
def analyze_cvs(uploaded_paths, expected_skills, target_field):
    results = []
    warnings = []
    progress_bar = st.progress(0)
    total_files = len(uploaded_paths)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_cv, path, expected_skills, target_field) for path in uploaded_paths]
        for i, future in enumerate(futures):
            result = future.result()
            if result:
                results.append(result)
            else:
                warnings.append(f"⚠️ CV tại {uploaded_paths[i]} không đạt tiêu chí và đã bị loại bỏ.")
            progress_bar.progress((i + 1) / total_files)

    if warnings:
        st.warning(f"⚠️ Có {len(warnings)} CV đã bị loại bỏ do không đạt tiêu chí.")
        with st.expander("Xem chi tiết các cảnh báo"):
            st.write("\n".join(warnings))

    return pd.DataFrame(results)

# --- Giao diện chính ---
def main():
    st.title("📄 Hệ thống Hỗ trợ Quản lý Tuyển dụng bằng AI")

    st.sidebar.header("📄 Upload CV")
    sample_cv_file = st.sidebar.file_uploader("📌 Tải lên CV tiêu chí", type="pdf")
    uploaded_files = st.sidebar.file_uploader("📅 Tải lên các CV ứng viên", type=["pdf"], accept_multiple_files=True)

    if sample_cv_file and uploaded_files:
        sample_cv_path = save_uploadedfile(sample_cv_file)
        sample_cv_text = extract_text_from_pdf(sample_cv_path)
        expected_skills = extract_skills_list(sample_cv_text)
        target_field = predict_field(sample_cv_text)

        uploaded_paths = [save_uploadedfile(uploaded_file) for uploaded_file in uploaded_files]
        st.sidebar.success(f"✅ Đã upload {len(uploaded_files)} CV ứng viên.")

        st.success("✅ Đang tiến hành phân tích CV...")
        my_bar = st.progress(0)

        df = analyze_cvs(uploaded_paths, expected_skills, target_field)
        my_bar.progress(1.0)

        st.subheader("📊 Tóm tắt kết quả")
        st.success(f"✅ Đã phân tích {len(df)} CV hợp lệ trên tổng số {len(uploaded_files)} CV.")

        if df.empty:
            st.warning("⚠️ Không có CV nào phù hợp với tiêu chí.")
        else:
            st.subheader("📋 Danh sách ứng viên phù hợp")
            df.index = df.index + 1
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📅 Tải danh sách ứng viên đã đánh giá",
                data=csv,
                file_name='cv_filtered_results.csv',
                mime='text/csv',
            )

            st.subheader("🔍 Xem chi tiết từng CV")
            selected_file = st.selectbox("Chọn một file CV để xem chi tiết:", df['Tên file'].tolist())

            if selected_file:
                selected_path = next((path for path in uploaded_paths if os.path.basename(path) == selected_file), None)
                if selected_path:
                    text = extract_text_from_pdf(selected_path)
                    if text:
                        st.write(f"### Phân tích chi tiết CV: {selected_file}")
                        st.write(f"- **Tên ứng viên**: {extract_name(text)}")

                        st.write("### Nội dung CV")
                        display_pdf(selected_path)

                        candidate_skills = extract_skills_list(text)
                        project_skills = extract_skills_from_projects(text)
                        total_skills = list(set(candidate_skills + project_skills))
                        matched, missing, skill_coverage = match_skills_accurately(total_skills, expected_skills)

                        st.write("### Tỉ lệ phù hợp")
                        st.write(f"- **Tổng**: {skill_coverage}%")

                        st.write("### Kỹ năng phù hợp")
                        st.write(", ".join(matched) if matched else "Không rõ")

                        st.write("### Kỹ năng còn thiếu")
                        st.write(", ".join(missing) if missing else "Không rõ")

                        st.write("### Kỹ năng trong project")
                        st.write(", ".join(project_skills) if project_skills else "Không rõ")

if __name__ == "__main__":
    main()
