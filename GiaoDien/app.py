import streamlit as st
import os
import pdfplumber
import re
import base64
from transformers import pipeline
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import spacy

# Cấu hình trang Streamlit.
st.set_page_config(page_title="Hệ thống Hỗ trợ Tuyển dụng bằng AI", layout="wide")

# Thư mục lưu file upload.
UPLOAD_FOLDER = './uploaded_cvs/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Tải mô hình zero-shot-classification và cache lại.
@st.cache_resource(show_spinner=True)
def load_classifier():
    try:
        return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình AI: {str(e)}. Vui lòng kiểm tra kết nối mạng hoặc thử lại sau.")
        return None

classifier = load_classifier()
FIELDS = ["Frontend Development", "Backend Development", "Data Science/AI", "DevOps", "Mobile Development"]

# Tải mô hình spaCy – dùng để nhận diện tên ứng viên.
@st.cache_resource(show_spinner=True)
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except Exception as e:
        st.error("Không tải được mô hình spaCy: " + str(e))
        return None

nlp_spacy = load_spacy_model()

############################################
# Các hàm xử lý file và trích xuất thông tin
############################################

def save_uploadedfile(uploadedfile):
    """Lưu file upload vào thư mục UPLOAD_FOLDER."""
    path = os.path.join(UPLOAD_FOLDER, uploadedfile.name)
    with open(path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return path

def extract_text_from_pdf(file_path):
    """Trích xuất text từ file PDF (chỉ lấy tối đa 3 trang đầu)."""
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

###############################
# CÁC HÀM TRÍCH XUẤT TÊN ỨNG VIÊN
###############################

def extract_name_first_line(text):
    """
    Bước 0: Nếu dòng đầu tiên của văn bản trông giống như tên ứng viên, 
    trả về dòng đầu tiên đó.
    Tiêu chí: không quá 50 ký tự, chứa ít nhất 2 từ và không có email hay số.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        first_line = lines[0]
        if len(first_line) <= 50 and len(first_line.split()) >= 2 and not re.search(r'\S+@\S+|\d', first_line):
            return first_line
    return None

def extract_name_spacy_full(text):
    """
    Chia toàn bộ văn bản thành các đoạn (chunk) với kích thước cố định và sử dụng spaCy
    để nhận diện tất cả các entity PERSON. Trả về candidate xuất hiện sớm nhất, 
    với điều kiện có ít nhất 2 từ và không chứa số.
    """
    chunk_size = 3000
    candidates = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        doc = nlp_spacy(chunk)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                candidate = ent.text.strip()
                if len(candidate.split()) >= 2 and not re.search(r'\d', candidate):
                    candidates.append((i + ent.start_char, candidate))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    return None

def extract_name_heuristic(text):
    """
    Quét toàn bộ các dòng trong văn bản và chọn tên ứng viên dựa trên các tiêu chí:
      - Độ dài dòng dưới 50 ký tự.
      - Ít nhất 2 từ.
      - Không chứa email hoặc số liệu.
      - Tỷ lệ từ in hoa >= 50%.
    Trả về candidate xuất hiện sớm nhất.
    """
    lines = text.splitlines()
    candidates = []
    for idx, line in enumerate(lines):
        line = line.strip()
        if len(line) > 50 or re.search(r'\S+@\S+|\d{3,}', line.lower()):
            continue
        words = line.split()
        if len(words) < 2:
            continue
        count_capitalized = sum(1 for w in words if w and w[0].isupper())
        if count_capitalized / len(words) >= 0.5:
            candidates.append((idx, line))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    return None

def extract_name_improved(text):
    """
    Cải thiện nhận diện tên ứng viên theo 4 bước:
      0. Kiểm tra dòng đầu tiên của văn bản.
      1. Dùng Regex: Tìm dòng chứa "Name:" hoặc "Tên:" và lấy phần sau dấu phân cách.
      2. Sử dụng spaCy trên toàn văn (chia thành các chunk) để nhận diện các entity PERSON.
      3. Nếu vẫn không tìm được, dùng fallback heuristic quét toàn bộ các dòng.
    Nếu không tìm được kết quả, trả về "Không rõ".
    """
    # Bước 0: Kiểm tra dòng đầu tiên.
    candidate_first = extract_name_first_line(text)
    if candidate_first:
        return candidate_first

    # Bước 1: Sử dụng Regex.
    regex_match = re.search(r'(Name|Tên)\s*[:\-]\s*(.+)', text, re.IGNORECASE)
    if regex_match:
        candidate = regex_match.group(2).split("\n")[0].strip()
        lower_candidate = candidate.lower()
        if candidate and not re.search(r'\S+@\S+|phone|\d', lower_candidate):
            return candidate

    # Bước 2: Sử dụng spaCy với chia chunk toàn văn.
    candidate_spacy = extract_name_spacy_full(text)
    if candidate_spacy:
        return candidate_spacy

    # Bước 3: Fallback heuristic quét toàn bộ các dòng.
    candidate_heuristic = extract_name_heuristic(text)
    if candidate_heuristic:
        return candidate_heuristic

    return "Không rõ"

############################################
# CÁC HÀM KHÁC (SKILL, PDF,...) 
############################################

def normalize_skill(skill):
    """Chuẩn hóa tên kỹ năng về dạng lowercase và loại bỏ ký tự thừa."""
    return skill.lower().replace("asp.net core", "asp.net").replace(".net core", ".net").replace("(", "").replace(")", "").strip()

def match_skills_accurately(candidate_skills, expected_skills):
    """
    So sánh danh sách kỹ năng của ứng viên với danh sách kỹ năng mong đợi.
    Trả về: danh sách kỹ năng trùng khớp, kỹ năng còn thiếu và % bao phủ.
    """
    norm_candidate = [normalize_skill(s) for s in candidate_skills]
    norm_expected = [normalize_skill(s) for s in expected_skills]
    matched = [s for s, norm_s in zip(expected_skills, norm_expected)
               if any(norm_s in c for c in norm_candidate)]
    missing = [s for s in expected_skills if s not in matched]
    coverage = round(len(matched) / len(expected_skills) * 100, 2) if expected_skills else 0
    return matched, missing, coverage

def extract_skills_list(text):
    """
    Trích xuất các kỹ năng từ các dòng chứa từ khóa như "skill", "tools", "tech", "technology", hoặc "framework".
    Sử dụng regex để tách các kỹ năng được liệt kê sau dấu phân cách.
    """
    skills = []
    for line in text.splitlines():
        if re.search(r'(skill|tools|tech|technology|framework)', line, re.IGNORECASE):
            parts = re.split(r'[:,]', line, maxsplit=1)
            if len(parts) > 1:
                items = re.split(r'[,/]', parts[1])
                items = [item.strip(" ()").strip() for item in items if item.strip()]
                skills.extend(items)
    return list(set(skills))

def extract_skills_list_improved(text):
    """
    Trích xuất các kỹ năng từ văn bản CV chỉ dựa trên nội dung của file,
    không bổ sung thêm từ danh sách kỹ năng cứng có sẵn.
    """
    return extract_skills_list(text)

def extract_skills_from_projects(text):
    """
    Trích xuất kỹ năng hoặc công nghệ từ các đoạn mô tả dự án có chứa các từ khóa như "project" hoặc "dự án".
    """
    sections = re.findall(r"(?i)(project|dự án)[^\n]*\n+(.*?)(?=\n{2,}|\Z)", text, re.DOTALL)
    all_skills = set()
    for _, section in sections:
        lines = section.split('\n')
        for line in lines:
            if any(kw in line.lower() for kw in ['stack', 'tech', 'technology', 'tools', 'framework', 'sử dụng']):
                items = re.split(r'[:,]', line, maxsplit=1)
                if len(items) > 1:
                    for part in re.split(r'[,/•]', items[1]):
                        skill = part.strip(" -•()")
                        if 1 < len(skill) <= 30:
                            all_skills.add(skill)
    return sorted(all_skills)

def predict_field(text_cv):
    """
    Sử dụng mô hình zero-shot để dự đoán mảng IT dựa trên nội dung của CV tiêu chí.
    """
    if classifier is None:
        return "Không xác định"
    short_text = text_cv[:1000]
    result = classifier(short_text, candidate_labels=FIELDS)
    return result['labels'][0]

def display_pdf(file_path):
    """Hiển thị file PDF trong Streamlit thông qua iframe."""
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Lỗi hiển thị PDF: {e}")

def process_cv(file_path, expected_skills, target_field):
    """
    Xử lý một file CV:
      - Trích xuất nội dung PDF.
      - Xác định tên ứng viên bằng extract_name_improved.
      - Trích xuất các kỹ năng từ toàn bộ CV và từ phần mô tả dự án.
      - So khớp danh sách kỹ năng với expected_skills và tính phần trăm bao phủ.
    """
    text = extract_text_from_pdf(file_path)
    if text:
        name = extract_name_improved(text)
        candidate_skills = extract_skills_list_improved(text)
        project_skills = extract_skills_from_projects(text)
        total_skills = list(set(candidate_skills + project_skills))
        matched, missing, skill_coverage = match_skills_accurately(total_skills, expected_skills)
        if skill_coverage == 0:
            return None
        result_status = "Phù hợp" if skill_coverage >= 50 else "Không phù hợp"
        return {
            'Tên file': os.path.basename(file_path),
            'Tên ứng viên': name,
            'Mảng IT': target_field,
            'Phần trăm phù hợp': skill_coverage,
            'Kết quả': result_status,
            'Kỹ năng phù hợp': ', '.join(matched),
            'Kỹ năng còn thiếu': ', '.join(missing),
            'Kỹ năng trong project': ', '.join(project_skills)
        }
    return None

@st.cache_data(show_spinner=True)
def analyze_cvs(uploaded_paths, expected_skills, target_field):
    """
    Phân tích các file CV sử dụng ThreadPoolExecutor để xử lý song song.
    Cập nhật progress bar và trả về kết quả dưới dạng DataFrame.
    """
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

############################################
# GIAO DIỆN CHÍNH cùng Streamlit
############################################

def main():
    st.title("📄 Hệ thống Hỗ trợ Quản lý Tuyển dụng bằng AI")
    st.sidebar.header("📄 Upload CV")
    
    sample_cv_file = st.sidebar.file_uploader("📌 Tải lên CV tiêu chí (PDF)", type="pdf")
    uploaded_files = st.sidebar.file_uploader("📅 Tải lên các CV ứng viên (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if sample_cv_file and uploaded_files:
        sample_cv_path = save_uploadedfile(sample_cv_file)
        sample_cv_text = extract_text_from_pdf(sample_cv_path)
        # Trích xuất kỹ năng từ CV tiêu chí dựa trên nội dung file.
        expected_skills = extract_skills_list_improved(sample_cv_text)
        target_field = predict_field(sample_cv_text)
    
        uploaded_paths = [save_uploadedfile(uploaded_file) for uploaded_file in uploaded_files]
        st.sidebar.success(f"✅ Đã upload {len(uploaded_files)} CV ứng viên.")
    
        st.success("✅ Đang tiến hành phân tích CV...")
        df = analyze_cvs(uploaded_paths, expected_skills, target_field)
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
                        st.write(f"- **Tên ứng viên**: {extract_name_improved(text)}")
                        st.write("### Nội dung CV")
                        display_pdf(selected_path)
                        candidate_skills = extract_skills_list_improved(text)
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