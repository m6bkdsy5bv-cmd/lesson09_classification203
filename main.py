import json
from pathlib import Path

import pandas as pd
import streamlit as st

TITLE = "뇌졸중 예측 실습실"
ICON = "🧠"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

SAVE_DIR = Path(__file__).parent / "내_기록"
SAVE_FILE = SAVE_DIR / "기록.json"

st.set_page_config(page_title=TITLE, page_icon=ICON, layout="wide")


# ---------- 저장 / 불러오기 ----------
def load_record():
    """내_기록/기록.json 을 읽어 온다. 없거나 깨졌으면 빈 기록을 돌려준다."""
    default = {"meanings": {}, "source": ""}
    try:
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        meanings = data.get("meanings", {})
        source = data.get("source", "")
        if not isinstance(meanings, dict) or not isinstance(source, str):
            return default
        return {"meanings": meanings, "source": source}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def save_record(record):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


# ---------- 데이터 ----------
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def describe_values(series):
    """열의 값 종류를 짧은 글로 설명한다."""
    values = series.dropna().unique()
    n = len(values)
    if n == 0:
        return "값 없음"
    if n <= 10:
        shown = ", ".join(str(v) for v in sorted(values, key=str))
        return f"{n}가지: {shown}"
    if pd.api.types.is_numeric_dtype(series):
        return f"숫자 (최소 {series.min():g} ~ 최대 {series.max():g}, {n}가지)"
    return f"{n}가지"


# ---------- 화면 ----------
st.markdown(
    f"<h1 style='margin-bottom:0'>{ICON} {TITLE}</h1>",
    unsafe_allow_html=True,
)
st.caption("뇌졸중 환자 데이터를 살펴보며 예측을 연습하는 곳")

st.markdown(
    """
    <style>
    .card {
        background: #f4f6fb;
        border-radius: 14px;
        padding: 22px 12px;
        text-align: center;
        border: 1px solid #e0e4f0;
    }
    .card .label { font-size: 1rem; color: #555; margin-bottom: 6px; }
    .card .value { font-size: 2.6rem; font-weight: 800; color: #1f3a8a; line-height: 1.1; }
    @media (prefers-color-scheme: dark) {
        .card { background: #1e2433; border-color: #2d3550; }
        .card .label { color: #b6bdd1; }
        .card .value { color: #8fb0ff; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결과 주소를 확인해 주세요.\n\n{e}")
    st.stop()

record = load_record()

st.header("1. 이 데이터는 무엇일까요?")

total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100 if total else 0

cards = [
    ("전체 사람 수", f"{total:,}명"),
    ("열 개수", f"{n_cols}개"),
    ("stroke가 1인 사람 수", f"{n_stroke:,}명"),
    ("stroke 1의 비율", f"{ratio:.2f}%"),
]
for col, (label, value) in zip(st.columns(4), cards):
    col.markdown(
        f"<div class='card'><div class='label'>{label}</div>"
        f"<div class='value'>{value}</div></div>",
        unsafe_allow_html=True,
    )

# ---------- 열 설명 표 ----------
st.subheader("열 설명표")
st.caption("‘우리말 뜻’ 칸을 눌러 직접 적어 보세요. 적은 내용은 자동으로 저장돼요.")

saved_meanings = record["meanings"]
table = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": [saved_meanings.get(c, "") for c in df.columns],
        "값의 종류": [describe_values(df[c]) for c in df.columns],
        "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
    }
)

edited = st.data_editor(
    table,
    key="column_table",
    hide_index=True,
    use_container_width=True,
    num_rows="fixed",
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻", width="medium"),
        "값의 종류": st.column_config.TextColumn("값의 종류", width="large"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d"),
    },
)

new_meanings = {
    row["열 이름"]: (row["우리말 뜻"] or "").strip() for _, row in edited.iterrows()
}
if new_meanings != {c: saved_meanings.get(c, "") for c in df.columns}:
    record["meanings"] = new_meanings
    save_record(record)

# ---------- 처음 다섯 줄 ----------
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

# ---------- 출처 ----------
st.subheader("데이터 출처")
source_text = st.text_area(
    "교재에 나온 출처를 적어 주세요. (입력 후 칸 밖을 누르면 저장돼요)",
    value=record["source"],
    height=120,
    key="source_box",
)
if source_text != record["source"]:
    record["source"] = source_text
    save_record(record)
    st.toast("출처를 저장했어요.", icon="✅")
