# 📦 Import thư viện
import streamlit as st
st.set_page_config(page_title="Hệ thống Hỗ trợ Tuyển dụng bằng AI", layout="wide")

import os
import pdfplumber
import re
import base64
from transformers import pipeline
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from collections import Counter
from streamlit_option_menu import option_menu

# --- CSS tuỳ chỉnh cho giao diện đẹp hơn ---
st.markdown("""
    <style>
    .stApp {
        background-color: #181c24;
    }
    .css-1d391kg, .css-1v0mbdj, .css-1cypcdb {
        color: #00d4ff !important;
    }
    .stSidebar {
        background: #23272f;
    }
    .sidebar-title {
        color: #00d4ff;
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 0px;
        margin-top: 10px;
        text-align: center;
        letter-spacing: 1px;
    }
    .sidebar-desc {
        color: #aaa;
        font-size: 14px;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        background: linear-gradient(90deg,#00d4ff,#1e90ff);
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }
    .stDownloadButton>button {
        background: linear-gradient(90deg,#00d4ff,#1e90ff);
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }
    .stDataFrame {
        background-color: #23272f;
    }
    .metric-label, .metric-value {
        color: #00d4ff !important;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(90deg,#00d4ff,#1e90ff);
    }
    </style>
""", unsafe_allow_html=True)

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

# --- Dashboard báo cáo ---
def show_dashboard(df):
    st.header("📊 Dashboard Báo cáo & Phân tích Kết quả")
    st.dataframe(df)

    # Thống kê tổng quan
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng số CV", len(df))
    col2.metric("Số CV phù hợp", (df['Kết quả'] == "Phù hợp").sum())
    col3.metric("Tỉ lệ phù hợp", f"{(df['Kết quả'] == 'Phù hợp').mean()*100:.1f}%")
    col4.metric("Số lĩnh vực", df['Mảng IT'].nunique())

    # Biểu đồ tỉ lệ phù hợp
    st.subheader("Tỉ lệ CV phù hợp/không phù hợp")
    status_counts = df['Kết quả'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    # Biểu đồ phân bố lĩnh vực IT
    st.subheader("Phân bố lĩnh vực IT")
    st.bar_chart(df['Mảng IT'].value_counts())

    # Top kỹ năng còn thiếu
    st.subheader("Top kỹ năng còn thiếu")
    missing_skills = []
    for skills in df['Kỹ năng còn thiếu']:
        missing_skills.extend([s.strip() for s in str(skills).split(',') if s.strip()])
    top_missing = Counter(missing_skills).most_common(10)
    if top_missing:
        skills, counts = zip(*top_missing)
        st.bar_chart(pd.Series(counts, index=skills))
    else:
        st.info("Không có kỹ năng còn thiếu nào nổi bật.")

    # Top kỹ năng phù hợp
    st.subheader("Top kỹ năng phù hợp")
    matched_skills = []
    for skills in df['Kỹ năng phù hợp']:
        matched_skills.extend([s.strip() for s in str(skills).split(',') if s.strip()])
    top_matched = Counter(matched_skills).most_common(10)
    if top_matched:
        skills, counts = zip(*top_matched)
        st.bar_chart(pd.Series(counts, index=skills))
    else:
        st.info("Không có kỹ năng phù hợp nổi bật.")

# --- Giao diện chính ---
def main():
    # Sidebar đẹp với option-menu và slogan
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
        st.markdown("<div class='sidebar-title'>Hệ thống Tuyển dụng AI</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-desc'>Tối ưu hóa quy trình tuyển dụng, lọc CV ứng viên tự động bằng AI.<br>Tiết kiệm thời gian, nâng cao hiệu quả!</div>", unsafe_allow_html=True)
        menu = option_menu(
            None,
            ["Phân tích CV", "Dashboard báo cáo"],
            icons=["file-earmark-text", "bar-chart"],
            menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#222"},
                "icon": {"color": "#00d4ff", "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"2px", "--hover-color": "#1e90ff"},
                "nav-link-selected": {"background-color": "#1e90ff", "color": "white"},
            }
        )

    st.title("📄 Hệ thống Hỗ trợ Quản lý Tuyển dụng bằng AI")

    if menu == "Phân tích CV":
        st.header("📄 Phân tích CV")
        st.markdown("> **Bước 1:** Tải lên CV tiêu chí (chuẩn).<br>**Bước 2:** Tải lên các CV ứng viên để lọc tự động.<br>**Bước 3:** Xem kết quả, tải danh sách ứng viên phù hợp.", unsafe_allow_html=True)
        sample_cv_file = st.file_uploader("📌 Tải lên CV tiêu chí", type="pdf")
        uploaded_files = st.file_uploader("📅 Tải lên các CV ứng viên", type=["pdf"], accept_multiple_files=True)

        if sample_cv_file and uploaded_files:
            sample_cv_path = save_uploadedfile(sample_cv_file)
            sample_cv_text = extract_text_from_pdf(sample_cv_path)
            expected_skills = extract_skills_list(sample_cv_text)
            target_field = predict_field(sample_cv_text)

            uploaded_paths = [save_uploadedfile(uploaded_file) for uploaded_file in uploaded_files]
            st.success(f"✅ Đã upload {len(uploaded_files)} CV ứng viên.")

            with st.spinner("🔎 Đang tiến hành phân tích CV..."):
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

            # Lưu DataFrame vào session_state để dùng cho dashboard nếu muốn
            st.session_state['last_df'] = df

    elif menu == "Dashboard báo cáo":
        st.header("📊 Dashboard Báo cáo & Phân tích Kết quả")
        st.markdown("> Tải lên file kết quả phân tích (CSV) hoặc sử dụng dữ liệu vừa phân tích để xem báo cáo tổng quan.", unsafe_allow_html=True)
        uploaded_csv = st.file_uploader("Tải lên file kết quả phân tích (CSV)", type="csv")
        df = None
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
        elif 'last_df' in st.session_state:
            df = st.session_state['last_df']
            st.info("Đang dùng dữ liệu kết quả vừa phân tích.")
        if df is not None and not df.empty:
            show_dashboard(df)
        else:
            st.info("Vui lòng tải lên file kết quả hoặc phân tích CV trước.")

if __name__ == "__main__":
    main()