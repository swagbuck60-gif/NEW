import streamlit as st
import pandas as pd
import re
from collections import defaultdict, Counter
from io import BytesIO

SHEET_TEACHER_WISE = "TEACHER WISE TIMETABLE"

DAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT"]
PERIOD_COLS = ["IST", "2ND", "3RD", "4TH", "5TH", "6TH", "7TH", "8TH"]

# -----------------------------
# Helpers to parse class codes
# -----------------------------

def normalize_class_code(code: str) -> str:
    """Normalize entries like 'XIA', 'IXA', 'X', 'VIA 1-3' -> leading class token."""
    if not isinstance(code, str):
        return ""
    s = code.strip().upper()
    if not s or s == "BREAK":
        return ""
    token = re.split(r"[ ,]", s)[0]
    return token

def class_to_band(cls: str) -> str:
    """Map class to band: senior (IX-XII), middle (VI-VIII), other."""
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
    """XI A/XII A=science, XI B/XII B=humanities."""
    if not cls:
        return ""
    c = cls.upper()
    if c.startswith("XIA") or c.startswith("XIIA"):
        return "science"
    if c.startswith("XIB") or c.startswith("XIIB"):
        return "humanities"
    return ""

# -----------------------------
# Teacher preference rules
# -----------------------------

def infer_designation(tname: str) -> str:
    s = tname.upper()
    if s.startswith("PGT"): return "PGT"
    if "PET" in s: return "PET"
    if "LIB" in s: return "LIB"
    if "MUSIC" in s: return "MUSIC"
    if "ART" in s: return "ART"
    if "TGT" in s: return "TGT"
    return "OTHER"

def is_light_duty_teacher(tname: str) -> bool:
    """Always available: TGT ART, TGT-MUSIC, PET-M, TGT-PET, PGT-PHYEDU, PET-F, COUNSELLOR."""
    s = tname.upper()
    light_duty_keywords = [
        "TGT ART", "TGT-MUSIC", "PET-M", "TGT-PET", 
        "PGT-PHYEDU", "PET-F", "COUNSELLOR"
    ]
    return any(keyword in s for keyword in light_duty_keywords)

def pgt_stream_role(tname: str) -> str:
    s = tname.upper()
    if s.startswith("PGT BIO") or s.startswith("PGT CHEM") or s.startswith("PGT PHY") or s.startswith("PGT MATH"):
        return "science"
    if any(kw in s for kw in ["PGT HIS", "PGT GEO", "PGT ECO"]):
        return "humanities"
    if any(kw in s for kw in ["PGT ENG", "PGT HIN", "PGT COM"]):
        return "common"
    return ""

def allowed_band_for_teacher(tname: str, teacher_bands):
    desg = infer_designation(tname)
    natural = teacher_bands.get(tname, set())
    if desg == "PGT": return {"senior"}
    if desg == "TGT": return {"middle", "senior"}
    if desg in {"PET", "MUSIC", "ART", "LIB"}:
        return natural if natural else {"middle", "senior"}
    return natural if natural else {"middle", "senior", "other"}

def score_candidate_for_slot(tname, band_needed, cls_needed, schedule, teacher_bands):
    bands_for_t = teacher_bands.get(tname, set())
    score = 100 if band_needed in bands_for_t else 0

    teaches_same_class = False
    teaches_band_count = 0
    for day, perdict in schedule[tname].items():
        for col, cls in perdict.items():
            if not cls: continue
            if normalize_class_code(cls) == cls_needed:
                teaches_same_class = True
            if class_to_band(cls) == band_needed:
                teaches_band_count += 1

    if teaches_same_class: score += 50
    score += min(teaches_band_count * 5, 50)

    cls_stream = stream_for_class(cls_needed)
    role = pgt_stream_role(tname)
    if cls_stream:
        if role == cls_stream: score += 120
        elif role == "common": score += 80
        elif role != "": score -= 100

    return score

# -----------------------------------
# Load TEACHER WISE timetable
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
            if i >= len(df): break
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
    schedule = defaultdict(lambda: defaultdict(dict))
    teacher_bands = defaultdict(set)
    for tname, tdf in teacher_tables.items():
        if "DAY" not in tdf.columns: continue
        for _, row in tdf.iterrows():
            day = str(row["DAY"]).strip().upper()
            if day not in DAYS: continue
            for col in PERIOD_COLS:
                if col not in tdf.columns: continue
                val = row[col]
                cls = "" if pd.isna(val) else normalize_class_code(str(val))
                schedule[tname][day][col] = cls
                if cls:
                    band = class_to_band(cls)
                    if band: teacher_bands[tname].add(band)
    return schedule, teacher_bands

# -----------------------------
# Arrangement generation - FINAL
# -----------------------------

@st.cache_data
def generate_arrangements(schedule, teacher_bands, day, absent_teachers, excluded_teachers,
                         include_teachers=None, max_arrangements_per_teacher=2):
    day = day.upper()
    results = []
    arrangement_count = Counter()

    # 1. Collect missing slots
    slots = []
    for t in absent_teachers:
        if t not in schedule: continue
        day_sched = schedule[t].get(day, {})
        for col in PERIOD_COLS:
            cls = day_sched.get(col, "")
            if cls:
                slots.append({
                    "day": day, "period_col": col, "class_code": cls,
                    "band": class_to_band(cls), "absent_teacher": t
                })

    # 2. Smart busy exclusion: OFF if >3 absent, light-duty always included
    busy_teachers = set()
    disable_busy_check = len(absent_teachers) > 3
    
    for tname in schedule.keys():
        if tname in absent_teachers or tname in excluded_teachers: continue
        
        # LIGHT-DUTY TEACHERS: Always available
        if is_light_duty_teacher(tname): continue
        
        # Apply 6+ exclusion only when ≤3 absent teachers
        if not disable_busy_check:
            day_sched = schedule[tname].get(day, {})
            busy_periods = sum(1 for col in PERIOD_COLS if day_sched.get(col, ""))
            if busy_periods >= 6:
                busy_teachers.add(tname)

    # 3. Process slots by period (no double-booking)
    slots_by_period = defaultdict(list)
    for slot in slots:
        slots_by_period[slot["period_col"]].append(slot)

    for period_col, period_slots in slots_by_period.items():
        available_teachers = set()
        for tname in schedule.keys():
            if (tname in absent_teachers or tname in excluded_teachers or 
                tname in busy_teachers or 
                (include_teachers is not None and tname not in include_teachers)):
                continue
            if schedule[tname].get(day, {}).get(period_col, ""): continue

            allowed_bands = allowed_band_for_teacher(tname, teacher_bands)
            can_teach_any = any(slot["band"] == "" or slot["band"] in allowed_bands 
                              for slot in period_slots)
            if can_teach_any:
                available_teachers.add(tname)

        for slot in period_slots:
            cls_needed, band_needed = slot["class_code"], slot["band"]
            candidates = []
            for tname in available_teachers:
                if arrangement_count[tname] >= max_arrangements_per_teacher: continue
                allowed_bands = allowed_band_for_teacher(tname, teacher_bands)
                if band_needed and band_needed not in allowed_bands: continue
                sc = score_candidate_for_slot(tname, band_needed, cls_needed, schedule, teacher_bands)
                candidates.append((tname, sc))

            if candidates:
                candidates.sort(key=lambda x: (-x[1], arrangement_count[x[0]]))
                tname, sc = candidates[0]
                arrangement_count[tname] += 1
                available_teachers.discard(tname)
                results.append({
                    "Day": day, "Period": period_col, "Class": cls_needed,
                    "Absent Teacher": slot["absent_teacher"], 
                    "Arrangement Teacher": tname, "Score": sc
                })
            else:
                results.append({
                    "Day": day, "Period": period_col, "Class": cls_needed,
                    "Absent Teacher": slot["absent_teacher"], 
                    "Arrangement Teacher": "NO SUITABLE TEACHER", "Score": 0
                })

    return pd.DataFrame(results)

# -----------------------------
# Streamlit UI - PRODUCTION READY
# -----------------------------

def main():
    st.set_page_config(page_title="JNV Arrangement Generator", layout="wide")
    st.title("🧑‍🏫 JNV Automatic Arrangement Generator")
    
    st.markdown("**Upload TEACHER WISE TIMETABLE** → Select absent staff → Get smart arrangements!")

    # File upload
    uploaded_file = st.file_uploader("📁 Upload final-school-timetable.xlsx", type="xlsx")
    if uploaded_file is None:
        st.info("👆 Please upload your **TEACHER WISE TIMETABLE** Excel file first.")
        st.stop()
    
    # Load data
    with st.spinner("Parsing teacher schedules..."):
        teacher_tables = load_teacher_wise(uploaded_file)
    
    if not teacher_tables:
        st.error("❌ No teacher data found in **TEACHER WISE TIMETABLE** sheet.")
        st.stop()

    schedule, teacher_bands = build_teacher_schedule(teacher_tables)
    all_teachers = sorted(teacher_tables.keys())

    # Day & stats
    st.subheader("1. Select day")
    col1, col2 = st.columns([3,1])
    with col1:
        day = st.selectbox("Day", DAYS, key="day")
    with col2:
        st.metric("Teachers loaded", len(all_teachers))

    # Absent teachers
    st.subheader("2. Select absent teachers")
    absent_teachers = st.multiselect("Absent today", all_teachers, key="absent")

    # Status indicator
    absent_count = len(absent_teachers)
    busy_status = "DISABLED (emergency mode)" if absent_count > 3 else "ACTIVE (protect busy teachers)"
    st.info(f"**{busy_status}** | ✅ TGT ART/MUSIC/PET always available")

    # Excluded teachers
    st.subheader("3. Exclude teachers (busy/duties)")
    excluded_teachers = st.multiselect(
        "Exclude from arrangements:", 
        [t for t in all_teachers if t not in absent_teachers], key="exclude")

    # Include teachers
    st.subheader("4. Restrict to specific teachers (optional)")
    include_teachers = st.multiselect(
        "Include **ONLY** these:", 
        [t for t in all_teachers if t not in absent_teachers and t not in excluded_teachers],
        help="Leave empty = all eligible teachers", key="include")

    # Settings
    st.subheader("5. Settings")
    col1, col2 = st.columns(2)
    with col1:
        max_arr = st.slider("Max arrangements per teacher", 0, 6, 2)
    with col2:
        st.info("**Light-duty teachers exempt from all limits**")

    # Generate
    if st.button("🚀 Generate Arrangements", type="primary"):
        if not absent_teachers:
            st.warning("⚠️ Select at least one absent teacher.")
            st.stop()
        
        with st.spinner("Finding optimal arrangements..."):
            df_arr = generate_arrangements(
                schedule, teacher_bands, day, absent_teachers,
                excluded_teachers, include_teachers, max_arr
            )
        
        if df_arr.empty:
            st.success("✅ No lost periods for selected teachers today.")
        else:
            st.success(f"✅ **{len(df_arr)} arrangements** generated!")
            st.dataframe(df_arr.style.highlight_max(subset=['Score'], axis=0), use_container_width=True)
            
            # Download
            buffer = BytesIO()
            df_arr.to_excel(buffer, index=False, engine="openpyxl")
            buffer.seek(0)
            st.download_button(
                label="📥 Download Excel",
                data=buffer,
                file_name=f"JNV_{day}_{absent_count}absent_arrangements.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    st.markdown("---")
    st.caption("✅ **JNV Baksa Optimized**: >3 absent=emergency mode | Light-duty always available | No double-booking")

if __name__ == "__main__":
    main()
