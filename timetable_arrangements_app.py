import streamlit as st
import pandas as pd
import re
from collections import defaultdict, Counter
from io import BytesIO

DAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT"]
PERIOD_COLS = ["IST", "2ND", "3RD", "4TH", "5TH", "6TH", "7TH", "8TH"]
SHEET_TEACHER_WISE = "TEACHER WISE TIMETABLE"

# -----------------------------
# Helpers to parse class codes
# -----------------------------

def normalize_class_code(code: str) -> str:
    """Normalize entries like 'XIA', 'IXA', 'X', 'VIA 1-3' -> leading class token (e.g., XIA, IXA, X, VIA)."""
    if not isinstance(code, str):
        return ""
    s = code.strip().upper()
    if not s or s == "BREAK":
        return ""
    token = re.split(r"[ ,]", s)[0]
    return token

def class_to_band(cls: str) -> str:
    """
    Map class token to band:
    - 'senior' : IX, X, XI, XII
    - 'middle' : VI, VII, VIII
    - 'other'  : anything else
    """
    if not cls:
        return ""
    m = re.match(r"(VI|VII|VIII|IX|X|XI|XII)", cls)
    if not m:
        return "other"
    r = m.group(1)
    if r in ["IX", "X", "XI", "XII"]:
        return "senior"
    if r in ["VI", "VII", "VIII"]:
        return "middle"
    return "other"

def stream_for_class(cls: str) -> str:
    """
    Return 'science', 'humanities', or '' for non XI/XII.
    XI A, XII A -> science
    XI B, XII B -> humanities
    """
    if not cls:
        return ""
    c = cls.upper()
    if c.startswith("XIA") or c.startswith("XIIA"):
        return "science"
    if c.startswith("XIB") or c.startswith("XIIB"):
        return "humanities"
    return ""

# -----------------------------------
# Load TEACHER WISE timetable blocks
# -----------------------------------

@st.cache_data(show_spinner=False)
def load_teacher_wise(uploaded_file):
    df = pd.read_excel(uploaded_file, sheet_name=SHEET_TEACHER_WISE, header=None)
    teacher_tables = {}
    i = 0
    while i < len(df):
        cell = df.iloc[i, 0]
        if isinstance(cell, str) and cell.strip() and "DAY" not in cell:
            teacher_name = cell.strip()
            i += 1
            if i >= len(df):
                break
            header_row = df.iloc[i].tolist()
            i += 1
            rows = []
            while i < len(df):
                first = df.iloc[i, 0]
                if isinstance(first, str) and first.strip() in DAYS:
                    rows.append(df.iloc[i].tolist())
                    i += 1
                else:
                    break
            while i < len(df) and (pd.isna(df.iloc[i, 0]) or str(df.iloc[i, 0]).strip() == ""):
                i += 1

            tdf = pd.DataFrame(rows, columns=[h if pd.notna(h) else "" for h in header_row])
            teacher_tables[teacher_name] = tdf
        else:
            i += 1
    return teacher_tables

# -----------------------------
# Build teacher schedule model
# -----------------------------

def build_teacher_schedule(teacher_tables):
    """
    Build structure:
    schedule[teacher][day][period] = class_token or ""
    teacher_bands[teacher] = set of bands naturally taught
    """
    schedule = defaultdict(lambda: defaultdict(dict))
    teacher_bands = defaultdict(set)
    for tname, tdf in teacher_tables.items():
        if "DAY" not in tdf.columns:
            continue
        for _, row in tdf.iterrows():
            day = str(row["DAY"]).strip().upper()
            if day not in DAYS:
                continue
            for col in PERIOD_COLS:
                if col not in tdf.columns:
                    continue
                val = row[col]
                if pd.isna(val):
                    cls = ""
                else:
                    cls = normalize_class_code(str(val))
                schedule[tname][day][col] = cls
                if cls:
                    band = class_to_band(cls)
                    if band:
                        teacher_bands[tname].add(band)
    return schedule, teacher_bands

# -----------------------------
# Teacher preference rules
# -----------------------------

def infer_designation(tname: str) -> str:
    s = tname.upper()
    if s.startswith("PGT"):
        return "PGT"
    if "PET" in s:
        return "PET"
    if "LIB" in s:
        return "LIB"
    if "MUSIC" in s:
        return "MUSIC"
    if "ART" in s:
        return "ART"
    if "TGT" in s:
        return "TGT"
    return "OTHER"

def pgt_stream_role(tname: str) -> str:
    """
    For XI/XII:
    - PGT BIO, PGT CHEM, PGT PHY, PGT MATHS -> science
    - PGT HIS, PGT GEO, PGT ECO           -> humanities
    - PGT ENG, PGT HIN, PGT COM           -> common
    """
    s = tname.upper()
    if s.startswith("PGT BIO") or s.startswith("PGT CHEM") or s.startswith("PGT PHY") or s.startswith("PGT MATH"):
        return "science"
    if "PGT  HIS" in s or "PGT HIS" in s or "PGT GEO" in s or "PGT ECO" in s:
        return "humanities"
    if "PGT ENG" in s or "PGT HIN" in s or "PGT COM" in s:
        return "common"
    return ""

def allowed_band_for_teacher(tname: str, teacher_bands):
    """
    Rules:
    - PGT: senior only (IX–XII)
    - TGT: middle + senior (VI–XII)
    - PET/MUSIC/ART/LIB: any of the bands they naturally teach; if none, middle+senior
    - OTHER: what they naturally teach, or all if unknown
    """
    desg = infer_designation(tname)
    natural = teacher_bands.get(tname, set())
    if desg == "PGT":
        return {"senior"}
    if desg == "TGT":
        return {"middle", "senior"}
    if desg in {"PET", "MUSIC", "ART", "LIB"}:
        if natural:
            return natural
        return {"middle", "senior"}
    if natural:
        return natural
    return {"middle", "senior", "other"}

def score_candidate_for_slot(tname, band_needed, cls_needed, schedule, teacher_bands):
    """
    Higher score = better candidate.
    - +100 if teacher naturally teaches this band.
    - +50 if teacher already teaches the same class code somewhere in week.
    - +up to 50 based on how often teacher teaches this band.
    - Stream-aware boost for XI/XII:
      * For XI A / XII A (science): prefer science PGTs, then common PGTs.
      * For XI B / XII B (humanities): prefer humanities PGTs, then common PGTs.
    """
    bands_for_t = teacher_bands.get(tname, set())
    score = 0

    # Band preference
    if band_needed in bands_for_t:
        score += 100

    teaches_same_class = False
    teaches_band_count = 0
    for day, perdict in schedule[tname].items():
        for col, cls in perdict.items():
            if not cls:
                continue
            if normalize_class_code(cls) == cls_needed:
                teaches_same_class = True
            if class_to_band(cls) == band_needed:
                teaches_band_count += 1

    if teaches_same_class:
        score += 50
    score += min(teaches_band_count * 5, 50)

    # Stream awareness for XI/XII
    cls_stream = stream_for_class(cls_needed)
    role = pgt_stream_role(tname)
    if cls_stream:
        if role == cls_stream:
            score += 120   # correct stream PGT
        elif role == "common":
            score += 80    # common PGTs next preference
        elif role != "":
            score -= 100   # wrong stream PGT discouraged

    return score

# -----------------------------
# Arrangement generation
# -----------------------------

def generate_arrangements(
    schedule,
    teacher_bands,
    day,
    absent_teachers,
    excluded_teachers,
    max_arrangements_per_teacher=2
):
    day = day.upper()
    results = []
    arrangement_count = Counter()

    # 1. Collect all missing slots
    slots = []
    for t in absent_teachers:
        if t not in schedule:
            continue
        day_sched = schedule[t].get(day, {})
        for col in PERIOD_COLS:
            cls = day_sched.get(col, "")
            if cls:
                band_needed = class_to_band(cls)
                slots.append({
                    "day": day,
                    "period_col": col,
                    "class_code": cls,
                    "band": band_needed,
                    "absent_teacher": t,
                })

    # 2. For each slot, choose best candidate
    for slot in slots:
        cls_needed = slot["class_code"]
        band_needed = slot["band"]
        col = slot["period_col"]

        candidates = []
        for tname in schedule.keys():
            if tname in absent_teachers:
                continue
            if tname in excluded_teachers:
                continue

            cls_here = schedule[tname].get(day, {}).get(col, "")
            if cls_here:
                continue

            allowed_bands = allowed_band_for_teacher(tname, teacher_bands)
            if band_needed and band_needed not in allowed_bands:
                continue

            sc = score_candidate_for_slot(tname, band_needed, cls_needed, schedule, teacher_bands)
            candidates.append((tname, sc))

        candidates.sort(key=lambda x: (-x[1], arrangement_count[x[0]]))

        assigned = None
        for tname, sc in candidates:
            if arrangement_count[tname] >= max_arrangements_per_teacher:
                continue
            assigned = (tname, sc)
            break

        if assigned:
            tname, sc = assigned
            arrangement_count[tname] += 1
            results.append({
                "Day": day,
                "Period": col,
                "Class": cls_needed,
                "Absent Teacher": slot["absent_teacher"],
                "Arrangement Teacher": tname,
                "Score": sc,
            })
        else:
            results.append({
                "Day": day,
                "Period": col,
                "Class": cls_needed,
                "Absent Teacher": slot["absent_teacher"],
                "Arrangement Teacher": "NO SUITABLE TEACHER",
                "Score": 0,
            })

    return pd.DataFrame(results)

# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.title("Automatic Daily Arrangement Generator")

    st.write(
        "Upload your **final-school-timetable.xlsx** file and automatically generate arrangement teachers for absent staff, "
        "respecting class bands (VI–VIII / IX–XII) and XI/XII streams."
    )

    # File uploader
    uploaded_file = st.file_uploader("Upload final-school-timetable.xlsx", type=["xlsx"])
    
    if uploaded_file is not None:
        with st.spinner("Loading teacher timetables..."):
            teacher_tables = load_teacher_wise(uploaded_file)
        
        if not teacher_tables:
            st.error("❌ No teacher data found in **TEACHER WISE TIMETABLE** sheet.")
            st.stop()

        schedule, teacher_bands = build_teacher_schedule(teacher_tables)
        all_teachers = sorted(teacher_tables.keys())

        st.success(f"✅ Loaded timetables for {len(all_teachers)} teachers")

        st.subheader("1. Select day")
        day = st.selectbox("Day", DAYS)

        st.subheader("2. Select absent teachers today")
        absent_teachers = st.multiselect("Absent teachers", all_teachers)

        st.subheader("3. Teachers to exclude from arrangements (busy / duties)")
        excluded_teachers = st.multiselect(
            "Exclude these teachers from being used for arrangements:",
            [t for t in all_teachers if t not in absent_teachers]
        )

        st.subheader("4. Arrangement settings")
        max_arr = st.slider("Maximum arrangements per teacher (today)", 0, 6, 2)

        if st.button("🚀 Generate arrangements", type="primary"):
            if not absent_teachers:
                st.warning("⚠️ Please select at least one absent teacher.")
            else:
                with st.spinner("Generating optimal arrangements..."):
                    df_arr = generate_arrangements(
                        schedule,
                        teacher_bands,
                        day,
                        absent_teachers,
                        excluded_teachers,
                        max_arrangements_per_teacher=max_arr
                    )
                
                if df_arr.empty:
                    st.info("ℹ️ No lost periods for the selected absent teachers on this day.")
                else:
                    st.success("✅ Arrangements generated successfully!")
                    st.dataframe(df_arr, use_container_width=True)

                    buffer = BytesIO()
                    df_arr.to_excel(buffer, index=False, engine="openpyxl")
                    buffer.seek(0)
                    st.download_button(
                        label="📥 Download arrangements as Excel",
                        data=buffer,
                        file_name=f"arrangements_{day}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

    else:
        st.info("👆 Please upload your **final-school-timetable.xlsx** file first")
        st.markdown("""
        ### 📋 Expected Excel Structure
        - **Sheet name**: `TEACHER WISE TIMETABLE`
        - **Teacher names** in column A (before DAY headers)
        - **Days**: MON, TUE, WED, THU, FRI, SAT
        - **Periods**: IST, 2ND, 3RD, 4TH, 5TH, 6TH, 7TH, 8TH
        """)

if __name__ == "__main__":
    main()
