import streamlit as st
import os
st.set_page_config(page_title="Hệ thống Hỗ trợ quản lý tuyển dụng ", layout="wide")

# --- Nhúng CSS từ file style.css ---
with open(os.path.join(os.path.dirname(__file__), "style.css"), encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
import plotly.graph_objects as go
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
        if re.search(r"(Name|Tên):", line, re.IGNORECASE):
            return line.split(":")[-1].strip()
    for line in lines[:10]:
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

# --- Trích xuất kỹ năng từ dự án (cải tiến) ---
def extract_skills_from_projects(text):
    project_skills = set()
    lines = text.split('\n')
    for i, line in enumerate(lines):
        line_lower = line.lower()
        # Nếu dòng chứa từ khóa kỹ năng
        if any(kw in line_lower for kw in ['tech stack', 'technology', 'tools', 'framework', 'sử dụng', 'environment', 'library', 'ngôn ngữ']):
            items = re.split(r"[:\-]", line, maxsplit=1)
            if len(items) > 1:
                for skill in re.split(r'[,/•;]', items[1]):
                    skill = skill.strip(" -•()")
                    if 1 < len(skill) <= 40:
                        project_skills.add(skill)
        else:
            # Nếu không có từ khóa, tìm kỹ năng phổ biến xuất hiện trong dòng mô tả dự án
            for skill in COMMON_SKILLS:
                if skill in line_lower:
                    project_skills.add(skill)
    return sorted(project_skills)

# --- Trích xuất tên dự án (cải tiến) ---
def extract_project_names(text):
    project_names = []
    lines = text.split('\n')
    for line in lines:
        line_strip = line.strip()
        # Loại bỏ các tiêu đề lớn
        if re.match(r"^(work|project|experience|notable|personal|projects?)\b", line_strip, re.IGNORECASE):
            continue
        # Nhận diện tên dự án dạng Project 1: Tên, Project: Tên, Dự án: Tên, ...
        if re.match(r"^(project\s*\d*|dự án|project name|tên dự án)\s*[:\-]", line_strip, re.IGNORECASE):
            name = re.split(r"[:\-]", line_strip, maxsplit=1)[-1].strip()
            if 3 < len(name) < 80:
                project_names.append(name)
        # Nhận diện các dòng có thể là tên dự án (viết hoa đầu, không quá dài, không có dấu chấm câu lớn)
        elif 3 < len(line_strip) < 80 and not any(x in line_strip.lower() for x in ["experience", "project", "work"]) and line_strip[0].isupper():
            project_names.append(line_strip)
    return list(dict.fromkeys(project_names))

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
        project_names = extract_project_names(text)
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
            'Dự án trong project': ', '.join(project_names),
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
            ["Phân tích CV", "So sánh CV", "Phần trăm phù hợp", "Dashboard báo cáo"],
            icons=["file-earmark-text", "files", "bar-chart-line", "bar-chart"],
            menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#222"},
                "icon": {"color": "#00d4ff", "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"2px", "--hover-color": "#1e90ff"},
                "nav-link-selected": {"background-color": "#1e90ff", "color": "white"},
            }
        )

    st.title("📄 Hệ thống hỗ trợ quản lý tuyển dụng ")

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
                st.session_state['last_df'] = None
                st.session_state['cv_valid_count'] = 0
                st.session_state['cv_invalid_count'] = 0
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
            if st.button("Xóa tất cả CV ứng viên", key="xoa_cv"):
                st.session_state['uploaded_paths'] = []
                st.session_state['last_df'] = None
                st.session_state['cv_valid_count'] = 0
                st.session_state['cv_invalid_count'] = 0
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

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 Danh sách ứng viên phù hợp"):
                    st.session_state['view_page'] = 'phuhop'
            with col2:
                if st.button("📋 Danh sách ứng viên không phù hợp", key="khongphuhop"):
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

    elif menu == "So sánh CV":
        st.header("🔍 So sánh CV")

        # Kiểm tra xem đã phân tích CV chưa
        if 'last_df' not in st.session_state or st.session_state['last_df'] is None or len(st.session_state['last_df']) == 0:
            st.warning("Vui lòng phân tích CV trước tại mục 'Phân tích CV'!")
        else:
            df = st.session_state['last_df']
            uploaded_paths = st.session_state['uploaded_paths']
            expected_skills = st.session_state['expected_skills']
            target_field = st.session_state['target_field']

            # Lọc danh sách CV phù hợp (phần trăm phù hợp >= 50%)
            suitable_df = df[df['Kết quả'] == "Phù hợp"]

            if len(suitable_df) == 0:
                st.warning("Không có CV nào phù hợp để so sánh. Vui lòng kiểm tra lại kết quả phân tích!")
            else:
                # Lựa chọn CV để so sánh từ danh sách CV phù hợp
                if 'selected_cvs' not in st.session_state:
                    st.session_state['selected_cvs'] = []

                selected_cvs = st.multiselect(
                    "Chọn CV phù hợp để so sánh:",
                    suitable_df['Tên file'].tolist(),
                    default=st.session_state['selected_cvs']
                )

                st.session_state['selected_cvs'] = selected_cvs

                if len(selected_cvs) < 2:
                    st.info("Vui lòng chọn ít nhất 2 CV để so sánh.")
                else:
                    # Tạo bảng so sánh
                    comparison_data = {
                        "Tiêu chí": [
                            "Tên ứng viên",
                            "Tên file",
                            "Mảng IT",
                            "Phần trăm phù hợp",
                            "Kết quả",
                            "Kỹ năng phù hợp",
                            "Kỹ năng còn thiếu",
                            "Kỹ năng trong project"
                        ]
                    }

                    # Lấy thông tin chi tiết của từng CV
                    cv_details = []
                    for selected_file in selected_cvs:
                        selected_path = next((path for path in uploaded_paths if os.path.basename(path) == selected_file), None)
                        if selected_path:
                            text = extract_text_from_pdf(selected_path)
                            if text:
                                candidate_name = extract_name(text)
                                candidate_skills = extract_skills_list(text)
                                project_skills = extract_skills_from_projects(text)
                                project_names = extract_project_names(text)
                                matched, missing, skill_coverage = match_skills_accurately(candidate_skills + project_skills, expected_skills, project_skills)
                                result = df.loc[df['Tên file'] == selected_file].iloc[0]

                                cv_details.append({
                                    'Tên file': selected_file,
                                    'Tên ứng viên': candidate_name,
                                    'Mảng IT': result['Mảng IT'],
                                    'Phần trăm phù hợp': result['Phần trăm phù hợp'],
                                    'Phần trăm phù hợp_raw': f"{result['Phần trăm phù hợp']}%",
                                    'Kết quả': result['Kết quả'],
                                    'Kỹ năng phù hợp': ', '.join(matched) if matched else 'Không rõ',
                                    'Kỹ năng còn thiếu': ', '.join(missing) if missing else 'Không rõ',
                                    'Kỹ năng trong project': ', '.join(project_skills) if project_skills else 'Không rõ',
                                    'Path': selected_path
                                })

                    # Điền dữ liệu vào bảng so sánh với highlight
                    for i, cv in enumerate(cv_details):
                        # Highlight Phần trăm phù hợp
                        percentage = cv['Phần trăm phù hợp']
                        percentage_str = cv['Phần trăm phù hợp_raw']
                        if percentage >= 50:
                            percentage_str = f"<div class='percentage-tooltip'><span class='highlight-suitable'>{percentage_str}</span><span class='tooltiptext'>Tỉ lệ kỹ năng phù hợp với yêu cầu công việc</span></div>"
                        else:
                            percentage_str = f"<div class='percentage-tooltip'><span class='highlight-unsuitable'>{percentage_str}</span><span class='tooltiptext'>Tỉ lệ kỹ năng phù hợp với yêu cầu công việc</span></div>"

                        # Highlight Kết quả
                        result = cv['Kết quả']
                        if result == "Phù hợp":
                            result = f"<span class='highlight-suitable'>{result}</span>"
                        else:
                            result = f"<span class='highlight-unsuitable'>{result}</span>"

                        # Highlight Kỹ năng phù hợp
                        matched_skills = cv['Kỹ năng phù hợp']
                        if matched_skills != "Không rõ":
                            matched_skills = f"<span class='highlight-skills-matched'>{matched_skills}</span>"

                        # Highlight Kỹ năng còn thiếu
                        missing_skills = cv['Kỹ năng còn thiếu']
                        if missing_skills != "Không rõ":
                            missing_skills = f"<span class='highlight-skills-missing'>{missing_skills}</span>"

                        comparison_data[f"CV {i+1}"] = [
                            cv['Tên ứng viên'],
                            cv['Tên file'],
                            cv['Mảng IT'],
                            percentage_str,
                            result,
                            matched_skills,
                            missing_skills,
                            cv['Kỹ năng trong project']
                        ]

                    # Hiển thị bảng so sánh
                    comparison_df = pd.DataFrame(comparison_data)

                    # Hàm xử lý xóa CV
                    def remove_cv(index):
                        if 0 <= index < len(st.session_state['selected_cvs']):
                            st.session_state['selected_cvs'].pop(index)
                        st.rerun()  # Sử dụng st.rerun() thay vì st.experimental_rerun()

                    # Thêm nút "Xóa" cho từng CV
                    st.subheader("📊 Bảng so sánh CV")
                    cols = st.columns([1] + [3] * len(selected_cvs))
                    with cols[0]:
                        st.write("")  # Cột đầu tiên để trống cho tiêu chí
                    for i, (cv, col) in enumerate(zip(cv_details, cols[1:])):
                        with col:
                            st.write(f"**CV {i+1}: {cv['Tên ứng viên']}**")
                            if st.button(f"Xóa CV {i+1}", key=f"remove_cv_{i}", help=f"Xóa CV {cv['Tên ứng viên']} khỏi bảng so sánh", on_click=lambda x=i: remove_cv(x)):
                                pass  # Logic xóa được xử lý trong remove_cv

                    # Thêm class CSS cho bảng
                    html_table = comparison_df.set_index("Tiêu chí").to_html(escape=False, classes="comparison-table")
                    st.markdown(html_table, unsafe_allow_html=True)

                    # Hiển thị CV gốc
                    st.subheader("📄 CV gốc của các ứng viên")
                    for cv in cv_details:
                        with st.expander(f"Xem CV gốc: {cv['Tên ứng viên']} ({cv['Tên file']})"):
                            display_pdf(cv['Path'])

    elif menu == "Phần trăm phù hợp":
        st.header("📊 Phần trăm phù hợp")

        # Kiểm tra xem đã phân tích CV chưa
        if 'last_df' not in st.session_state or st.session_state['last_df'] is None or len(st.session_state['last_df']) == 0:
            st.warning("Vui lòng phân tích CV trước tại mục 'Phân tích CV'!")
        else:
            df = st.session_state['last_df']

            # Lọc danh sách CV phù hợp (phần trăm phù hợp >= 50%)
            suitable_df = df[df['Kết quả'] == "Phù hợp"]

            if len(suitable_df) == 0:
                st.warning("Không có CV nào phù hợp để so sánh. Vui lòng kiểm tra lại kết quả phân tích!")
            else:
                # Lựa chọn nhiều CV để so sánh từ danh sách CV phù hợp
                selected_files = st.multiselect("Chọn các CV phù hợp để xem tỷ lệ phần trăm:", suitable_df['Tên file'].tolist(), default=suitable_df['Tên file'].tolist()[:2], max_selections=5)

                if len(selected_files) < 1:
                    st.info("Vui lòng chọn ít nhất 1 CV để xem biểu đồ.")
                else:
                    # Tạo DataFrame chứa các CV được chọn
                    comparison_df = suitable_df[suitable_df['Tên file'].isin(selected_files)][['Tên ứng viên', 'Phần trăm phù hợp']]
                    comparison_df = comparison_df.reset_index(drop=True)
                    comparison_df.index = range(1, len(comparison_df) + 1)

                    # Hiển thị biểu đồ cột bằng Plotly
                    st.subheader("📈 So sánh phần trăm phù hợp")
                    fig = go.Figure(data=
                        go.Bar(
                            x=comparison_df['Tên ứng viên'],
                            y=comparison_df['Phần trăm phù hợp'],
                            marker_color=['#00d4ff', '#1e90ff', '#00b7eb', '#007bff', '#00aaff'][:len(selected_files)],
                            text=comparison_df['Phần trăm phù hợp'],
                            textposition='auto'
                        )
                    )
                    fig.update_layout(
                        title='So sánh phần trăm phù hợp',
                        xaxis_title="Tên ứng viên",
                        yaxis_title="Phần trăm phù hợp (%)",
                        yaxis_range=[0, 100],
                        plot_bgcolor='#181c24',
                        paper_bgcolor='#181c24',
                        font_color='#00d4ff'
                    )
                    st.plotly_chart(fig)

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