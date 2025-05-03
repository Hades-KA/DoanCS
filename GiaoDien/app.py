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

# 📥 Thư mục lưu CV
UPLOAD_FOLDER = './uploaded_cvs/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 📚 Load mô hình AI
@st.cache_resource
def load_classifier():
    try:
        return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình AI: {str(e)}. Vui lòng kiểm tra kết nối mạng hoặc thử lại sau.")
        return None

classifier = load_classifier()

FIELDS = ["Frontend Development", "Backend Development", "Data Science/AI", "DevOps", "Mobile Development"]

# --- Hàm lưu file ---
def save_uploadedfile(uploadedfile):
    """Lưu file được tải lên."""
    path = os.path.join(UPLOAD_FOLDER, uploadedfile.name)
    with open(path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return path

# --- Hàm xử lý ---
def extract_text_from_pdf(file_path):
    """Trích xuất văn bản từ file PDF (giới hạn 3 trang đầu)."""
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages[:3]:  # Chỉ đọc 3 trang đầu tiên
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Lỗi khi đọc file PDF: {str(e)}")
        return ""

def extract_name(text):
    """Trích xuất tên ứng viên từ văn bản."""
    lines = text.strip().split("\n")
    for line in lines[:15]:  # Kiểm tra 15 dòng đầu tiên
        if re.search(r"(Name|Tên):", line, re.IGNORECASE):  # Tìm từ khóa "Name" hoặc "Tên"
            return line.split(":")[-1].strip()  # Lấy phần sau dấu ":"
        if len(line.split()) >= 2 and line[0].isupper():  # Giả định tên có ít nhất 2 từ và chữ cái đầu viết hoa
            if not any(char.isdigit() for char in line):  # Loại bỏ các dòng chứa số
                return line.strip()
    return "Không rõ"

def predict_field(text_cv):
    """Dự đoán ngành nghề từ nội dung CV."""
    if classifier is None:
        return "Không xác định"
    short_text = text_cv[:300]  # Giới hạn 300 ký tự đầu tiên
    result = classifier(short_text, candidate_labels=FIELDS)
    return result['labels'][0]

def extract_skills(text):
    """Trích xuất kỹ năng từ văn bản."""
    skills = [
        "html", "css", "javascript", "react", "vue.js", "angular",  # Frontend
        "node.js", "django", "spring boot", "flask", "sql", "mysql", "postgresql",  # Backend
        "python", "machine learning", "deep learning", "pandas", "tensorflow", "keras",  # Data Science/AI
        "docker", "kubernetes", "aws", "ci/cd", "jenkins",  # DevOps
        "flutter", "react native", "swift", "kotlin"  # Mobile
    ]
    found_skills = [skill for skill in skills if skill.lower() in text.lower()]
    return found_skills

def analyze_skills(cv_text, required_skills):
    """Phân tích kỹ năng hiện có, còn thiếu và kỹ năng khác."""
    cv_skills = extract_skills(cv_text)
    key_skills = [skill for skill in cv_skills if skill in required_skills]
    missing_skills = [skill for skill in required_skills if skill not in cv_skills]
    other_skills = [skill for skill in cv_skills if skill not in required_skills]
    skills_coverage = round(min(len(key_skills) / len(required_skills) * 100, 100), 2) if required_skills else 0
    return key_skills, missing_skills, other_skills, skills_coverage

def calculate_soft_skills_score(text):
    """Tính điểm kỹ năng mềm dựa trên từ khóa."""
    soft_skills_keywords = [
        "teamwork", "communication", "leadership", "problem solving", "critical thinking",
        "adaptability", "creativity", "time management", "collaboration", "empathy"
    ]
    found_skills = [skill for skill in soft_skills_keywords if skill.lower() in text.lower()]
    score = round((len(found_skills) / len(soft_skills_keywords)) * 100, 2)
    return score

def display_pdf(file_path):
    """Hiển thị file PDF trong giao diện Streamlit."""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def process_cv(file_path, sample_cv_text):
    """Xử lý một CV."""
    text = extract_text_from_pdf(file_path)
    if text:
        predicted_field = predict_field(text)
        name = extract_name(text)
        skills = extract_skills(text)
        required_skills = extract_skills(sample_cv_text)
        key_skills, missing_skills, other_skills, skills_coverage = analyze_skills(text, required_skills)

        # Loại bỏ CV không có kỹ năng chuyên môn
        if not key_skills:
            return None

        hard_skills_score = skills_coverage
        soft_skills_score = calculate_soft_skills_score(text)  # Tính điểm kỹ năng mềm
        overall_score = round((0.6 * hard_skills_score + 0.4 * soft_skills_score), 2)
        result_status = "Phù hợp" if overall_score >= 50 else "Không phù hợp"

        return {
            'Tên file': os.path.basename(file_path),
            'Tên ứng viên': name,
            'Mảng IT': predicted_field,
            'Kỹ năng chuyên môn': hard_skills_score,
            'Kỹ năng mềm': soft_skills_score,
            'Điểm tổng': overall_score,
            'Kết quả': result_status,
            'Kỹ năng còn thiếu': ', '.join(missing_skills)
        }
    return None

@st.cache_data
def analyze_cvs(uploaded_paths, sample_cv_text):
    """Phân tích tất cả các CV song song."""
    results = []
    warnings = []  # Danh sách lưu cảnh báo
    progress_bar = st.progress(0)  # Thanh tiến trình duy nhất
    total_files = len(uploaded_paths)

    with ThreadPoolExecutor(max_workers=5) as executor:  # Giới hạn tối đa 5 luồng xử lý
        futures = [executor.submit(process_cv, path, sample_cv_text) for path in uploaded_paths]
        for i, future in enumerate(futures):
            result = future.result()
            if result:
                results.append(result)
            else:
                warnings.append(f"⚠️ CV tại {uploaded_paths[i]} không có kỹ năng chuyên môn và đã bị loại bỏ.")
            progress_bar.progress((i + 1) / total_files)  # Cập nhật tiến trình tổng thể

    # Hiển thị cảnh báo sau khi hoàn tất
    if warnings:
        st.warning(f"⚠️ Có {len(warnings)} CV đã bị loại bỏ do không có kỹ năng chuyên môn.")
        with st.expander("Xem chi tiết các cảnh báo"):
            st.write("\n".join(warnings))  # Hiển thị tất cả cảnh báo trong một khối cuộn

    return pd.DataFrame(results)

# --- Giao diện chính ---
def main():
    st.title("📄 Hệ thống Hỗ trợ Quản lý Tuyển dụng bằng AI")

    st.sidebar.header("📤 Upload CV (.pdf)")
    uploaded_files = st.sidebar.file_uploader("Tải lên nhiều CV", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        uploaded_paths = [save_uploadedfile(uploaded_file) for uploaded_file in uploaded_files]
        st.sidebar.success(f"✅ Đã upload {len(uploaded_files)} file CV.")

        st.success("✅ Đang tiến hành phân tích CV...")
        my_bar = st.progress(0)

        sample_cv_text = extract_text_from_pdf('./data/cv_samples/CV_FE_PhamNgocVienDong.pdf')
        df = analyze_cvs(uploaded_paths, sample_cv_text)
        my_bar.progress(1.0)

        # Hiển thị tóm tắt kết quả
        st.subheader("📊 Tóm tắt kết quả")
        st.success(f"✅ Đã phân tích {len(df)} CV hợp lệ trên tổng số {len(uploaded_files)} CV.")
        st.warning(f"⚠️ {len(uploaded_files) - len(df)} CV đã bị loại bỏ.")

        if df.empty:
            st.warning("⚠️ Không có CV nào được phân tích.")
        else:
            st.subheader("📋 Danh sách ứng viên ngành IT")
            df.index = df.index + 1  # Đánh số thứ tự từ 1
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Tải danh sách ứng viên đã đánh giá",
                data=csv,
                file_name='cv_evaluation_results.csv',
                mime='text/csv',
            )

            st.subheader("🔍 Xem và phân tích chi tiết CV")
            selected_file = st.selectbox("Chọn một file CV để xem chi tiết:", df['Tên file'].tolist())

            if selected_file:
                selected_path = next((path for path in uploaded_paths if os.path.basename(path) == selected_file), None)
                if selected_path:
                    text = extract_text_from_pdf(selected_path)
                    if text:
                        st.write(f"### Phân tích chi tiết CV: {selected_file}")
                        st.write(f"- **Tên ứng viên**: {extract_name(text)}")

                        # Hiển thị file PDF
                        st.write("### Nội dung CV")
                        display_pdf(selected_path)

                        required_skills = extract_skills(sample_cv_text)
                        key_skills, missing_skills, other_skills, skills_coverage = analyze_skills(text, required_skills)

                        st.write("### Phân tích kỹ năng")
                        st.write(f"- **Kỹ năng hiện có ({len(key_skills)}):** {', '.join(key_skills)}")
                        st.write(f"- **Kỹ năng còn thiếu ({len(missing_skills)}):** {', '.join(missing_skills)}")
                        st.write(f"- **Kỹ năng khác ({len(other_skills)}):** {', '.join(other_skills)}")
                        st.write(f"- **Phần trăm bao phủ kỹ năng:** {skills_coverage}%")

if __name__ == "__main__":
    main()