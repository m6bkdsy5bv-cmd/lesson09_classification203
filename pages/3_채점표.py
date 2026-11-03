import numpy as np
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
RANDOM_STATE = 42
MAX_ROWS = 20
NA = "계산할 수 없음"

st.set_page_config(page_title=f"채점표 · {TITLE}", page_icon="📝", layout="wide")

# '분류 모델' 페이지에서 처음 골라져 있는 입력 속성과 같다.
FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]
KOR = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

NAME_LR = "로지스틱 회귀(확률로 답하는 모델)"
NAME_DT = "의사결정트리(질문으로 답하는 모델)"
NAME_BASE = "훈련용에서 많은 쪽으로만 답하는 모델"


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def pct(x):
    return f"{x * 100:.2f}%"


def count_cells(y_true, y_pred):
    """뇌졸중(1)을 양성으로 두고 네 칸을 센다."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def confusion_table(tp, fp, fn, tn):
    return pd.DataFrame(
        [
            [f"TP · {tp:,}명", f"FN · {fn:,}명", f"{tp + fn:,}명"],
            [f"FP · {fp:,}명", f"TN · {tn:,}명", f"{fp + tn:,}명"],
            [f"{tp + fp:,}명", f"{fn + tn:,}명", f"{tp + fp + fn + tn:,}명"],
        ],
        index=["실제 뇌졸중", "실제 아님", "합계"],
        columns=["예측 뇌졸중", "예측 아님", "합계"],
    )


def four_metrics(tp, fp, fn, tn):
    """정확도·재현율·정밀도·F1을 글자로 돌려준다. 분모가 0이면 '계산할 수 없음'."""
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None  # 분모: 실제 뇌졸중인 사람 수
    precision = tp / (tp + fp) if (tp + fp) > 0 else None  # 분모: 뇌졸중이라 예측한 사람 수
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    def show(v):
        return NA if v is None else pct(v)

    return {
        "정확도": show(accuracy),
        "재현율": show(recall),
        "정밀도": show(precision),
        "F1": show(f1),
    }


# ---------- 화면 맨 위 ----------
st.markdown("<h1 style='margin-bottom:0'>📝 채점표</h1>", unsafe_allow_html=True)
st.caption(
    "테스트 데이터로 두 모델의 답을 채점해요. 뇌졸중(stroke=1)을 양성으로 두고, "
    "입력 속성은 '분류 모델' 페이지의 처음 설정(나이·평균 혈당·고혈압·심장병)과 같아요."
)

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결과 주소를 확인해 주세요.\n\n{e}")
    st.stop()

# ---------- 테스트용 / 훈련용 나누기 ('분류 모델' 페이지와 같은 방법) ----------
# 번호(id) 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정한다.
data = df.sort_values("id").reset_index(drop=True)
is_test = (np.arange(len(data)) % 10) < 3
train_df = data[~is_test]
test_df = data[is_test].reset_index(drop=True)

X_train = train_df[FEATURES].copy()
X_test = test_df[FEATURES].copy()
y_train = train_df["stroke"]
y_test = test_df["stroke"].to_numpy()

# ---------- 크기 맞추기: 훈련용으로만 배우고 두 데이터에 똑같이 적용 ----------
scaler = StandardScaler().fit(X_train)
Xs_train = scaler.transform(X_train)
Xs_test = scaler.transform(X_test)

# ---------- 모델 만들기 ----------
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(Xs_train, y_train)
dt = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE
).fit(X_train, y_train)
base = DummyClassifier(strategy="most_frequent").fit(Xs_train, y_train)

predictions = {
    NAME_LR: lr.predict(Xs_test),
    NAME_DT: dt.predict(X_test),
    NAME_BASE: base.predict(Xs_test),
}
cells = {name: count_cells(y_test, p) for name, p in predictions.items()}

n_test = len(test_df)
n_test_stroke = int((y_test == 1).sum())
st.write(f"테스트용 {n_test:,}명 가운데 뇌졸중인 사람은 {n_test_stroke:,}명이에요.")

# ---------- 1. 혼동행렬 ----------
st.header("1. 혼동행렬")
st.caption(
    "행은 실제, 열은 예측이에요. TP는 뇌졸중을 맞힌 사람, FN은 뇌졸중을 놓친 사람, "
    "FP는 아닌데 뇌졸중이라고 한 사람, TN은 아님을 맞힌 사람이에요."
)
left, right = st.columns(2)
for col, name in zip((left, right), (NAME_LR, NAME_DT)):
    with col:
        st.subheader(name)
        st.table(confusion_table(*cells[name]))

# ---------- 2. 지표 비교표 ----------
st.header("2. 네 가지 지표 비교")
rows = {name: four_metrics(*cells[name]) for name in (NAME_LR, NAME_DT, NAME_BASE)}
metric_table = pd.DataFrame.from_dict(rows, orient="index")
metric_table.index.name = "모델"
st.table(metric_table)
st.caption(
    "정확도 = (TP + TN) ÷ 전체 사람 수 · "
    "재현율 = TP ÷ (TP + FN), 분모는 실제 뇌졸중인 사람 수 · "
    "정밀도 = TP ÷ (TP + FP), 분모는 뇌졸중이라 예측한 사람 수 · "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율). "
    "분모가 0이면 '계산할 수 없음'이라고 적어요."
)

# ---------- 3. 놓친 환자 / 헛짚은 사람 ----------
st.header("3. 놓친 환자와 헛짚은 사람")
choice = st.radio(
    "어느 모델의 목록을 볼까요?", [NAME_LR, NAME_DT], horizontal=True
)
pred = predictions[choice]
tp, fp, fn, tn = cells[choice]


def person_table(mask):
    """나이가 많은 순으로 정렬하고, 앞 MAX_ROWS명만 표로 만든다."""
    sub = test_df[mask].sort_values(["age", "id"], ascending=[False, True])
    shown = sub.head(MAX_ROWS)
    table = pd.DataFrame(
        {
            "번호": shown["id"].to_numpy(),
            "나이": shown["age"].to_numpy(),
            "평균 혈당": shown["avg_glucose_level"].to_numpy(),
            "고혈압": np.where(shown["hypertension"].to_numpy() == 1, "있음", "없음"),
            "심장병": np.where(shown["heart_disease"].to_numpy() == 1, "있음", "없음"),
        }
    )
    return table, len(sub)


missed_table, n_missed = person_table((y_test == 1) & (pred == 0))
wrong_table, n_wrong = person_table((y_test == 0) & (pred == 1))

left, right = st.columns(2)
for col, title, desc, table, n in (
    (left, "놓친 환자", "실제 뇌졸중인데 아니라고 예측한 사람", missed_table, n_missed),
    (right, "헛짚은 사람", "아닌데 뇌졸중이라고 예측한 사람", wrong_table, n_wrong),
):
    with col:
        st.subheader(f"{title} (전체 {n:,}명)")
        caption = desc
        if n > MAX_ROWS:
            caption += f" · 나이가 많은 순으로 앞 {MAX_ROWS}명만 보여요."
        else:
            caption += " · 나이가 많은 순이에요."
        st.caption(caption)
        if n == 0:
            st.info("해당하는 사람이 없어요.")
        else:
            st.dataframe(table, hide_index=True)

same = (n_missed == fn) and (n_wrong == fp)
if same:
    st.write(
        f"두 목록의 사람 수(놓친 환자 {n_missed:,}명, 헛짚은 사람 {n_wrong:,}명)는 "
        f"{choice}의 혼동행렬 FN {fn:,}명, FP {fp:,}명 칸과 같아요."
    )
else:
    st.warning(
        f"두 목록의 사람 수(놓친 환자 {n_missed:,}명, 헛짚은 사람 {n_wrong:,}명)가 "
        f"혼동행렬 FN {fn:,}명, FP {fp:,}명 칸과 달라요. 코드를 확인해야 해요."
    )
