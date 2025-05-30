import streamlit as st
st.set_page_config(page_title="Hệ thống Hỗ trợ Tuyển dụng bằng AI", layout="wide")

import os
import pdfplumber
import re
import base64
from transformers import pipeline
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from rapidfuzz import fuzz
from streamlit_option_menu import option_menu
import unicodedata
import matplotlib.pyplot as plt
from collections import Counter

# --- Danh sách kỹ năng lập trình phổ biến ---
COMMON_SKILLS = [
    "javascript", "typescript", "reactjs", "redux", "tailwindcss", "java", "spring boot", "spring", "spring framework",
    "spring security", "spring jpa", "validate", "mysql", "sql server", "antd", "cloudinary", "jwt", "php", "vuejs",
    "html", "css", "nodejs", "python", "docker", "kubernetes", "aws", "azure", "flask", "django", "c#", "c++", "android",
    "ios", "react native", "swift", "kotlin"
]

def extract_present_skills(text):
    text_lower = text.lower()
    present_skills = []
    for skill in COMMON_SKILLS:
        if skill in text_lower and skill not in present_skills:
            present_skills.append(skill)
    return present_skills

# --- CSS tuỳ chỉnh ---
st.markdown("""
    <style>
    .stApp { background-color: #181c24; }
    .css-1d391kg, .css-1v0mbdj, .css-1cypcdb { color: #00d4ff !important; }
    .stSidebar { background: #23272f; }
    .sidebar-title { color: #00d4ff; font-size: 22px; font-weight: bold; text-align: center; }
    .sidebar-desc { color: #aaa; font-size: 14px; text-align: center; }
    .stButton>button, .stDownloadButton>button { background: linear-gradient(90deg,#00d4ff,#1e90ff); color: white; }
    .stDataFrame { background-color: #23272f; }
    .metric-label, .metric-value { color: #00d4ff !important; }
    .stProgress > div > div > div > div { background-image: linear-gradient(90deg,#00d4ff,#1e90ff); }
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
        st.error(f"Lỗi khi tải mô hình AI: {str(e)}. Vui lòng kiểm tra kết nối mạng hoặc thử lại sau.")
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
            for page in pdf.pages[:3]:  # Chỉ đọc 3 trang đầu
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
        for line in lines[:30]:
            if re.search(r"(Name|Tên):", line, re.IGNORECASE):
                return line.split(":")[-1].strip()
    for line in lines[:10]:
        for line in lines[:30]:
            if len(line.split()) >= 2 and line[0].isupper():
                if not any(char.isdigit() for char in line) and len(line.split()) <= 5 and not any(kw in line.lower() for kw in ["contact", "information"]):
                    if not any(char.isdigit() for char in line) and len(line.split()) <= 5:
                        return line.strip()
    return "Không rõ"

# --- Phân loại lĩnh vực, trả về cả score ---
def predict_field(text_cv):
    if classifier is None:
        return "Không xác định", 0.0
    short_text = text_cv[:300]
    result = classifier(short_text, candidate_labels=FIELDS)
    return result['labels'][0], result['scores'][0]

# --- Trích xuất kỹ năng từ CV (dựa trên danh sách phổ biến) ---
def extract_skills_list(text):
    text_lower = text.lower()
    skills = []
    for skill in COMMON_SKILLS:
        if skill in text_lower and skill not in skills:
            skills.append(skill)
    return skills

# --- Trích xuất kỹ năng từ dự án ---
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
def normalize_skill(skill):
    skill = unicodedata.normalize('NFKD', skill).encode('ASCII', 'ignore').decode('utf-8')
    return skill.lower().strip()

def match_skills_accurately(candidate_skills, expected_skills, project_skills):
    candidate_skills = list(set(normalize_skill(skill) for skill in candidate_skills))
    expected_skills = list(set(normalize_skill(skill) for skill in expected_skills))
    project_skills = list(set(normalize_skill(skill) for skill in project_skills))

    matched = []
    for expected in expected_skills:
        for candidate in candidate_skills:
            if (
                fuzz.ratio(expected, candidate) >= 50
                or expected in candidate
                or candidate in expected
            ):
                matched.append(expected)
                break

    matched = list(set(matched))
    missing = [s for s in expected_skills if s not in matched]
    missing = [s for s in missing if s not in project_skills]
    missing = list(set(missing))
    coverage = round(len(matched) / len(expected_skills) * 100, 2) if expected_skills else 0
    return matched, missing, coverage

# --- Kiểm tra lĩnh vực ---
def match_field(text, target_field):
    text = text.lower()
    target_field = target_field.lower()
    field_keywords = {
        "frontend development": ["frontend", "html", "css", "javascript", "react", "angular", "vue"],
        "backend development": ["backend", "node.js", "django", "flask", "spring", "java", "php"],
        "data science/ai": ["data science", "machine learning", "ai", "deep learning", "pandas", "numpy"],
        "devops": ["devops", "docker", "kubernetes", "ci/cd", "aws", "azure", "cloud"],
        "mobile development": ["mobile", "android", "ios", "flutter", "react native", "swift", "kotlin"]
    }
    keywords = field_keywords.get(target_field, [])
    return any(keyword in text for keyword in keywords)

# --- Hiển thị file PDF ---
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
        matched, missing, skill_coverage = match_skills_accurately(total_skills, expected_skills, project_skills)
        present_skills = extract_present_skills(text)
        result_status = "Phù hợp" if skill_coverage >= 50 else "Không phù hợp"
        return {
            'Tên file': os.path.basename(file_path),
            'Tên ứng viên': name,
            'Mảng IT': target_field,
            'Phần trăm phù hợp': skill_coverage,
            'Kết quả': result_status,
            'Kỹ năng hiện có': ', '.join(present_skills),
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

    # --- Khởi tạo session_state ---
    if 'last_df' not in st.session_state:
        st.session_state['last_df'] = None
    if 'uploaded_paths' not in st.session_state:
        st.session_state['uploaded_paths'] = []
    if 'expected_skills' not in st.session_state:
        st.session_state['expected_skills'] = []
    if 'target_field' not in st.session_state:
        st.session_state['target_field'] = ""
    if 'sample_cv_path' not in st.session_state:
        st.session_state['sample_cv_path'] = None
    if 'cv_valid_count' not in st.session_state:
        st.session_state['cv_valid_count'] = 0
    if 'cv_invalid_count' not in st.session_state:
        st.session_state['cv_invalid_count'] = 0
    if 'view_page' not in st.session_state:
        st.session_state['view_page'] = 'phuhop'

    if menu == "Phân tích CV":
        st.header("📄 Phân tích CV")

        # --- Xử lý upload CV tiêu chí ---
        if st.session_state['sample_cv_path']:
            st.success(f"Đã upload CV tiêu chí: {os.path.basename(st.session_state['sample_cv_path'])}")
            if st.button("Xóa CV tiêu chí"):
                st.session_state['sample_cv_path'] = None
                st.session_state['expected_skills'] = []
                st.session_state['target_field'] = ""
        else:
            sample_cv_file = st.file_uploader("📌 Tải lên CV tiêu chí", type="pdf", key="sample_cv_file")
            if sample_cv_file:
                st.session_state['sample_cv_path'] = save_uploadedfile(sample_cv_file)
                sample_cv_text = extract_text_from_pdf(st.session_state['sample_cv_path'])
                st.session_state['expected_skills'] = extract_skills_list(sample_cv_text)
                st.session_state['target_field'], _ = predict_field(sample_cv_text)

        # --- Xử lý upload các CV ứng viên ---
        if st.session_state['uploaded_paths']:
            st.success(f"Đã upload {len(st.session_state['uploaded_paths'])} CV ứng viên.")
            if st.button("Xóa tất cả CV ứng viên"):
                st.session_state['uploaded_paths'] = []
        else:
            uploaded_files = st.file_uploader("📅 Tải lên các CV ứng viên", type=["pdf"], accept_multiple_files=True, key="uploaded_files")
            if uploaded_files:
                st.session_state['uploaded_paths'] = [save_uploadedfile(f) for f in uploaded_files]

        # --- Hiển thị lại thông báo kết quả ---
        if st.session_state['cv_valid_count'] > 0:
            st.success(f"✅ Đã phân tích {st.session_state['cv_valid_count']} CV hợp lệ trên tổng số {st.session_state['cv_valid_count'] + st.session_state['cv_invalid_count']} CV.")
        if st.session_state['cv_invalid_count'] > 0:
            st.warning(f"⚠️ Có {st.session_state['cv_invalid_count']} CV đã bị loại bỏ do không đạt tiêu chí.")

        # --- Phân tích khi đủ dữ liệu ---
        if st.session_state['sample_cv_path'] and st.session_state['uploaded_paths']:
            if st.button("🚀 Phân tích CV ứng viên"):
                expected_skills = st.session_state['expected_skills']
                target_field = st.session_state['target_field']
                uploaded_paths = st.session_state['uploaded_paths']

                with st.spinner("🔎 Đang tiến hành phân tích CV..."):
                    my_bar = st.progress(0)
                    df = analyze_cvs(uploaded_paths, expected_skills, target_field)
                    my_bar.progress(1.0)

                st.session_state['last_df'] = df
                st.session_state['cv_valid_count'] = len(df)
                st.session_state['cv_invalid_count'] = len(uploaded_paths) - len(df)

        # --- Hiển thị kết quả nếu đã phân tích ---
        if st.session_state['last_df'] is not None and len(st.session_state['last_df']) > 0:
            df = st.session_state['last_df']
            uploaded_paths = st.session_state['uploaded_paths']
            expected_skills = st.session_state['expected_skills']
            target_field = st.session_state['target_field']

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 Danh sách ứng viên phù hợp"):
                    st.session_state['view_page'] = 'phuhop'
            with col2:
                if st.button("📋 Danh sách ứng viên không phù hợp"):
                    st.session_state['view_page'] = 'khongphuhop'

            if st.session_state['view_page'] == 'phuhop':
                df_show = df[df['Kết quả'] == "Phù hợp"]
                st.subheader(f"✅ Danh sách ứng viên phù hợp ({len(df_show)})")
            else:
                df_show = df[df['Kết quả'] == "Không phù hợp"]
                st.subheader(f"❌ Danh sách ứng viên không phù hợp ({len(df_show)})")

            if df_show.empty:
                st.warning("Không có ứng viên trong danh sách này.")
            else:
                df_show = df_show.copy()
                df_show.index = range(1, len(df_show) + 1)
                st.dataframe(df_show)

                st.subheader("🔍 Xem chi tiết từng CV")
                selected_file = st.selectbox("Chọn một file CV để xem chi tiết:", df_show['Tên file'].tolist())

                if selected_file:
                    selected_path = next((path for path in uploaded_paths if os.path.basename(path) == selected_file), None)
                    if selected_path:
                        text = extract_text_from_pdf(selected_path)
                        if text:
                            st.markdown(f"### 📄 Phân tích chi tiết CV: `{selected_file}`")
                            display_pdf(selected_path)

                            st.markdown(
                                """
                                <div style="background-color: #1e293b; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
                                    <p><strong>Tên file:</strong> {}</p>
                                    <p><strong>Tên ứng viên:</strong> {}</p>
                                    <p><strong>Mảng IT:</strong> {}</p>
                                    <p><strong>Phần trăm phù hợp:</strong> {}%</p>
                                    <p><strong>Kết quả:</strong> {}</p>
                                </div>
                                """.format(
                                    selected_file,
                                    extract_name(text),
                                    target_field,
                                    df_show.loc[df_show['Tên file'] == selected_file, 'Phần trăm phù hợp'].values[0],
                                    df_show.loc[df_show['Tên file'] == selected_file, 'Kết quả'].values[0]
                                ),
                                unsafe_allow_html=True
                            )

                            present_skills = extract_present_skills(text)
                            st.markdown("### 🛠️ Kỹ năng CV hiện có")
                            st.markdown(
                                "<ul>" + "".join(f"<li>{skill}</li>" for skill in present_skills) + "</ul>"
                                if present_skills else "Không rõ",
                                unsafe_allow_html=True
                            )

                            candidate_skills = extract_skills_list(text)
                            project_skills = extract_skills_from_projects(text)
                            total_skills = list(set(candidate_skills + project_skills))
                            matched, missing, skill_coverage = match_skills_accurately(total_skills, expected_skills, project_skills)

                            st.markdown("### 📊 Tỉ lệ phù hợp")
                            st.markdown(f"- **Tổng**: {skill_coverage}%")

                            st.markdown("### ✅ Kỹ năng phù hợp")
                            st.markdown(
                                "<ul>" + "".join(f"<li>{skill}</li>" for skill in matched) + "</ul>"
                                if matched else "Không rõ",
                                unsafe_allow_html=True
                            )

                            st.markdown("### ❌ Kỹ năng còn thiếu")
                            st.markdown(
                                "<ul>" + "".join(f"<li>{skill}</li>" for skill in missing) + "</ul>"
                                if missing else "Không rõ",
                                unsafe_allow_html=True
                            )

                            st.markdown("### 📂 Kỹ năng trong project")
                            st.markdown(
                                "<ul>" + "".join(f"<li>{skill}</li>" for skill in project_skills) + "</ul>"
                                if project_skills else "Không rõ",
                                unsafe_allow_html=True
                            )

    elif menu == "Dashboard báo cáo":
        st.header("📊 Dashboard Báo cáo & Phân tích Kết quả")
        st.markdown("> Tải lên file kết quả phân tích (CSV) hoặc sử dụng dữ liệu vừa phân tích để xem báo cáo tổng quan.", unsafe_allow_html=True)
        uploaded_csv = st.file_uploader("Tải lên file kết quả phân tích (CSV)", type="csv")
        df = None
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
        elif st.session_state['last_df'] is not None:
            df = st.session_state['last_df']
            st.info("Đang dùng dữ liệu kết quả vừa phân tích.")

        if df is not None and not df.empty:
            st.subheader("📋 Dữ liệu phân tích CV")
            st.dataframe(df)

            total_cv = len(df)
            suitable_cv = len(df[df['Kết quả'] == "Phù hợp"])
            unsuitable_cv = total_cv - suitable_cv
            avg_skill_coverage = df['Phần trăm phù hợp'].mean()

            st.subheader("📊 Thống kê tổng quan")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Tổng số CV", total_cv)
            col2.metric("Số CV phù hợp", suitable_cv)
            col3.metric("Số CV không phù hợp", unsuitable_cv)
            col4.metric("Tỉ lệ kỹ năng phù hợp TB", f"{avg_skill_coverage:.2f}%")

            st.subheader("📈 Phân bố tỉ lệ phù hợp")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(df['Phần trăm phù hợp'], bins=10, color='skyblue', edgecolor='black')
            ax.set_title("Phân bố tỉ lệ phù hợp")
            ax.set_xlabel("Tỉ lệ phù hợp (%)")
            ax.set_ylabel("Số lượng CV")
            st.pyplot(fig)

            st.subheader("🛠️ Kỹ năng phổ biến trong CV")
            all_skills = []
            for skills in df['Kỹ năng hiện có']:
                if isinstance(skills, str):
                    all_skills.extend([s.strip() for s in skills.split(",") if s.strip()])
            skill_counts = Counter(all_skills)
            skill_df = pd.DataFrame(skill_counts.items(), columns=["Kỹ năng", "Số lượng"]).sort_values(by="Số lượng", ascending=False)
            st.bar_chart(skill_df.set_index("Kỹ năng"))

            st.subheader("❌ Kỹ năng còn thiếu phổ biến")
            all_missing_skills = []
            for skills in df['Kỹ năng còn thiếu']:
                if isinstance(skills, str):
                    all_missing_skills.extend([s.strip() for s in skills.split(",") if s.strip()])
            missing_skill_counts = Counter(all_missing_skills)
            missing_skill_df = pd.DataFrame(missing_skill_counts.items(), columns=["Kỹ năng", "Số lượng"]).sort_values(by="Số lượng", ascending=False)
            st.write(missing_skill_df)

            st.subheader("📥 Tải xuống dữ liệu")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Tải xuống file CSV",
                data=csv,
                file_name="ket_qua_phan_tich_cv.csv",
                mime="text/csv"
            )
        else:
            st.info("Vui lòng tải lên file kết quả hoặc phân tích CV trước.")

if __name__ == "__main__":
    main()