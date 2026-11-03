import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
RANDOM_STATE = 42

st.set_page_config(page_title=f"분류 모델 · {TITLE}", page_icon="🤖", layout="wide")

FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
KOR = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
BINARY = {"hypertension", "heart_disease"}
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

NAME_LR = "로지스틱 회귀(확률로 답하는 모델)"
NAME_DT = "의사결정트리(질문으로 답하는 모델)"
NAME_BASE = "입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델"

COLOR_POS = "#d62728"  # 뇌졸중
COLOR_NEG = "#1f77b4"  # 뇌졸중 아님
PALE_NEG = "#dbeaf7"
PALE_POS = "#fbdcdc"
LEAF_NEG = "#cfe3f7"
LEAF_POS = "#f7cfcf"


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.markdown(
    """
    <style>
    .card {
        background: #f4f6fb;
        border-radius: 14px;
        padding: 18px 14px;
        text-align: center;
        border: 1px solid #e0e4f0;
        height: 100%;
    }
    .card .name { font-size: 0.95rem; color: #555; margin-bottom: 8px; min-height: 2.6em; }
    .card .big { font-size: 2.8rem; font-weight: 800; color: #1f3a8a; line-height: 1.1; }
    .card .biglabel { font-size: 0.85rem; color: #777; margin-bottom: 10px; }
    .card .small { display: flex; justify-content: space-around; font-size: 0.85rem; color: #555; }
    @media (prefers-color-scheme: dark) {
        .card { background: #1e2433; border-color: #2d3550; }
        .card .name, .card .small { color: #b6bdd1; }
        .card .big { color: #8fb0ff; }
        .card .biglabel { color: #98a0b8; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def pct(x):
    return f"{x * 100:.2f}%"


def card_html(name, train_acc, test_acc):
    return (
        "<div class='card'>"
        f"<div class='name'>{name}</div>"
        f"<div class='big'>{pct(test_acc)}</div>"
        "<div class='biglabel'>테스트 데이터의 정확도</div>"
        "<div class='small'>"
        f"<span>훈련 {pct(train_acc)}</span>"
        f"<span>테스트 {pct(test_acc)}</span>"
        "</div></div>"
    )


def fmt_value(feature, value):
    if feature in BINARY:
        if value == 0:
            return f"{KOR[feature]} 0(없음)"
        if value == 1:
            return f"{KOR[feature]} 1(있음)"
        return f"{KOR[feature]} {value:g}"
    return f"{KOR[feature]} {value:.1f}"


# ---------- 화면 맨 위 ----------
st.markdown("<h1 style='margin-bottom:0'>🤖 분류 모델</h1>", unsafe_allow_html=True)
st.caption("뇌졸중(stroke=1)을 양성으로 두고, 두 가지 모델이 뇌졸중 여부를 얼마나 맞히는지 비교해요.")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결과 주소를 확인해 주세요.\n\n{e}")
    st.stop()

# ---------- 입력 속성 고르기 ----------
chosen = st.multiselect(
    "입력으로 사용할 속성",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda f: KOR[f],
)
selected = [f for f in FEATURES if f in chosen]

if len(selected) < 2:
    st.info("입력 속성을 두 개 이상 골라 주세요. 그래야 모델을 만들고 그림을 그릴 수 있어요.")
    st.stop()

# ---------- 테스트용 / 훈련용 나누기 ----------
# 번호(id) 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정한다.
data = df.sort_values("id").reset_index(drop=True)
is_test = (np.arange(len(data)) % 10) < 3
train_df = data[~is_test]
test_df = data[is_test]

X_train = train_df[selected].copy()
X_test = test_df[selected].copy()
y_train = train_df["stroke"]
y_test = test_df["stroke"]

st.caption(
    f"번호 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정했어요. "
    f"테스트용 {len(test_df):,}명, 훈련용 {len(train_df):,}명이에요."
)

# ---------- 빈 값 채우기 (체질량지수를 고른 경우에만, 훈련용 중앙값) ----------
if "bmi" in selected:
    bmi_median = X_train["bmi"].median()
    n_fill_train = int(X_train["bmi"].isna().sum())
    n_fill_test = int(X_test["bmi"].isna().sum())
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
    st.caption(
        f"체질량지수가 비어 있던 사람(훈련용 {n_fill_train:,}명, 테스트용 {n_fill_test:,}명)은 "
        f"훈련용의 중앙값 {bmi_median:.1f}로 채웠어요."
    )

# ---------- 크기 맞추기: 훈련용으로만 배우고 두 데이터에 똑같이 적용 ----------
scaler = StandardScaler().fit(X_train)
Xs_train = scaler.transform(X_train)
Xs_test = scaler.transform(X_test)

# ---------- 모델 만들기 ----------
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(Xs_train, y_train)
# 의사결정트리: 질문은 세 번까지, 마지막 마디에는 사람이 다섯 명 이상, 난수 고정
dt = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE
).fit(X_train, y_train)
base = DummyClassifier(strategy="most_frequent").fit(Xs_train, y_train)

acc = {
    "lr": (
        accuracy_score(y_train, lr.predict(Xs_train)),
        accuracy_score(y_test, lr.predict(Xs_test)),
    ),
    "dt": (
        accuracy_score(y_train, dt.predict(X_train)),
        accuracy_score(y_test, dt.predict(X_test)),
    ),
    "base": (
        accuracy_score(y_train, base.predict(Xs_train)),
        accuracy_score(y_test, base.predict(Xs_test)),
    ),
}

# ---------- 1. 정확도 카드 ----------
st.header("1. 정확도 비교")
for col, (key, name) in zip(
    st.columns(3), [("lr", NAME_LR), ("dt", NAME_DT), ("base", NAME_BASE)]
):
    col.markdown(card_html(name, acc[key][0], acc[key][1]), unsafe_allow_html=True)
st.caption(
    "의사결정트리(질문으로 답하는 모델)는 질문을 세 번까지만 던지고, 마지막 마디에는 "
    "사람이 다섯 명 이상 남도록 했어요. 난수는 고정했어요(42). "
    "이 모델은 크기를 맞추지 않은 원래 값으로 질문을 던져서, 아래 가지 그림의 기준값을 그대로 읽을 수 있어요."
)

# ---------- 2. 산점도 ----------
st.header("2. 테스트 데이터 산점도와 두 모델이 나눈 선·칸")
c1, c2 = st.columns(2)
x_feat = c1.selectbox("가로축", selected, index=0, format_func=lambda f: KOR[f])
y_options = [f for f in selected if f != x_feat]
y_feat = c2.selectbox("세로축", y_options, index=0, format_func=lambda f: KOR[f])


def axis_range(col):
    lo, hi = float(X_test[col].min()), float(X_test[col].max())
    span = hi - lo
    if col in BINARY:
        pad = 0.25
    else:
        pad = span * 0.05 if span > 0 else 1.0
    return lo - pad, hi + pad


x0, x1 = axis_range(x_feat)
y0, y1 = axis_range(y_feat)
xs = np.linspace(x0, x1, 160)
ys = np.linspace(y0, y1, 160)
XX, YY = np.meshgrid(xs, ys)

others = [f for f in selected if f not in (x_feat, y_feat)]
other_medians = {f: float(X_test[f].median()) for f in others}

grid = pd.DataFrame({x_feat: XX.ravel(), y_feat: YY.ravel()})
for f in others:
    grid[f] = other_medians[f]
grid = grid[selected]

prob = lr.predict_proba(scaler.transform(grid))[:, 1].reshape(XX.shape)
tree_pred = dt.predict(grid).reshape(XX.shape)

fig = go.Figure()

# 의사결정트리가 나눈 칸 (옅은 색)
fig.add_trace(
    go.Heatmap(
        x=xs,
        y=ys,
        z=tree_pred,
        zmin=0,
        zmax=1,
        colorscale=[[0, PALE_NEG], [1, PALE_POS]],
        showscale=False,
        hoverinfo="skip",
    )
)

# 테스트 데이터 점
rng = np.random.default_rng(0)
px_vals = X_test[x_feat].to_numpy(dtype=float)
py_vals = X_test[y_feat].to_numpy(dtype=float)
jittered = False
if x_feat in BINARY:
    px_vals = px_vals + rng.uniform(-0.12, 0.12, len(px_vals))
    jittered = True
if y_feat in BINARY:
    py_vals = py_vals + rng.uniform(-0.12, 0.12, len(py_vals))
    jittered = True

for label, value, color in [
    ("실제로 뇌졸중이 아님", 0, COLOR_NEG),
    ("실제로 뇌졸중", 1, COLOR_POS),
]:
    m = (y_test == value).to_numpy()
    fig.add_trace(
        go.Scatter(
            x=px_vals[m],
            y=py_vals[m],
            mode="markers",
            name=label,
            marker=dict(color=color, size=6 if value == 1 else 5, opacity=0.65),
        )
    )

# 로지스틱 회귀가 0.5로 가르는 선
line_inside = bool(prob.min() < 0.5 < prob.max())
if line_inside:
    fig.add_trace(
        go.Contour(
            x=xs,
            y=ys,
            z=prob,
            contours=dict(start=0.5, end=0.5, size=1, coloring="lines", showlabels=False),
            line=dict(color="black", width=3),
            showscale=False,
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="로지스틱 회귀 경계선(뇌졸중 확률 0.5)",
            line=dict(color="black", width=3),
        )
    )

# 범례: 트리 칸 색
present = set(np.unique(tree_pred).astype(int).tolist())
for cls, label, color in [
    (0, "트리가 '아님'이라고 답하는 칸", PALE_NEG),
    (1, "트리가 '뇌졸중'이라고 답하는 칸", PALE_POS),
]:
    if cls in present:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=label,
                marker=dict(symbol="square", size=14, color=color, line=dict(color="#999", width=1)),
            )
        )

fig.update_layout(
    height=580,
    xaxis=dict(title=KOR[x_feat], range=[x0, x1]),
    yaxis=dict(title=KOR[y_feat], range=[y0, y1]),
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0),
    margin=dict(t=30),
)
st.plotly_chart(fig, use_container_width=True)

# 그림 아래 글
if others:
    fixed_text = ", ".join(fmt_value(f, other_medians[f]) for f in others)
    st.write(f"그림에 쓰지 않은 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {fixed_text}")
else:
    st.write("고른 속성이 두 개뿐이라, 세워 두고 계산한 속성은 없어요.")

if line_inside:
    st.write(
        "검은 선이 로지스틱 회귀(확률로 답하는 모델)가 뇌졸중 확률 0.5로 가르는 선이에요. "
        "선을 사이에 두고 한쪽은 뇌졸중, 다른 쪽은 아님이라고 답해요."
    )
elif prob.max() <= 0.5:
    st.write(
        "로지스틱 회귀(확률로 답하는 모델)가 0.5로 가르는 선은 이 그림 밖에 있어요. "
        f"그림 안 모든 자리에서 뇌졸중 확률이 0.5보다 낮고, 가장 높은 곳도 {prob.max() * 100:.1f}%예요."
    )
else:
    st.write(
        "로지스틱 회귀(확률로 답하는 모델)가 0.5로 가르는 선은 이 그림 밖에 있어요. "
        f"그림 안 모든 자리에서 뇌졸중 확률이 0.5보다 높고, 가장 낮은 곳도 {prob.min() * 100:.1f}%예요."
    )

if len(present) == 1:
    only = "뇌졸중" if 1 in present else "아님"
    st.write(f"의사결정트리(질문으로 답하는 모델)는 이 그림 안 모든 칸에서 '{only}'이라고 답해요.")

notes = []
if jittered:
    notes.append("0과 1만 있는 속성은 점이 겹치지 않게 살짝 흩어서 찍었어요.")
if "bmi" in selected and "bmi" in (x_feat, y_feat):
    notes.append("비어 있던 체질량지수는 훈련용 중앙값으로 채운 값으로 찍혀 있어요.")
if notes:
    st.caption(" ".join(notes))

# ---------- 3. 의사결정트리 가지 그림 ----------
st.header("3. 의사결정트리(질문으로 답하는 모델)가 던진 질문")

t = dt.tree_
path = dt.decision_path(X_train)
n_all = np.asarray(path.sum(axis=0)).ravel()
pos_rows = np.flatnonzero((y_train.to_numpy() == 1))
n_pos = np.asarray(path[pos_rows].sum(axis=0)).ravel()

is_leaf = t.children_left == -1
answers = t.value[:, 0, :].argmax(axis=1)  # 0: 아님, 1: 뇌졸중


def fmt_thr(x):
    return f"{x:.4g}"


dot = [
    "digraph Tree {",
    'graph [rankdir=TB, nodesep=0.3, ranksep=0.5];',
    'node [shape=box, style="rounded,filled", fontsize=12, margin="0.12,0.08"];',
    "edge [fontsize=12];",
]
for i in range(t.node_count):
    stats = (
        f"훈련용 {n_all[i]:,}명\\n뇌졸중 {n_pos[i]:,}명 "
        f"({n_pos[i] / n_all[i] * 100:.1f}%)"
    )
    if is_leaf[i]:
        ans = "뇌졸중" if answers[i] == 1 else "아님"
        fill = LEAF_POS if answers[i] == 1 else LEAF_NEG
        dot.append(f'n{i} [label="답: {ans}\\n{stats}", fillcolor="{fill}"];')
    else:
        feat = selected[t.feature[i]]
        if feat in BINARY:
            question = f"{KOR[feat]} 있음?"
            left_label, right_label = "아니요", "예"
        else:
            question = f"{KOR[feat]} ≤ {fmt_thr(t.threshold[i])}?"
            left_label, right_label = "예", "아니요"
        dot.append(f'n{i} [label="{question}\\n{stats}", fillcolor="#ffffff"];')
        dot.append(f'n{i} -> n{t.children_left[i]} [label="{left_label}"];')
        dot.append(f'n{i} -> n{t.children_right[i]} [label="{right_label}"];')
dot.append("}")

st.graphviz_chart("\n".join(dot), use_container_width=True)

n_leaves = int(is_leaf.sum())
n_leaf_neg = int((is_leaf & (answers == 0)).sum())
used_idx = set(int(f) for f in t.feature[~is_leaf])
used = [f for f in selected if selected.index(f) in used_idx]
unused = [f for f in selected if f not in used]

st.write(
    f"답을 내는 마디는 모두 {n_leaves}칸이고, 그중 {n_leaf_neg}칸이 '아님'이라고 답해요."
)
if used:
    extra = f" (묻지 않은 속성: {', '.join(KOR[f] for f in unused)})" if unused else ""
    st.write(f"고른 속성 가운데 이 나무가 실제로 물은 것: {', '.join(KOR[f] for f in used)}{extra}")
else:
    st.write("고른 속성 가운데 이 나무가 실제로 물은 것은 없어요. 질문 없이 바로 답을 내요.")
