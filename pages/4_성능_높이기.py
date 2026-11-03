import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
RANDOM_STATE = 42
NA = "계산할 수 없음"

st.set_page_config(page_title=f"성능 높이기 · {TITLE}", page_icon="⚖️", layout="wide")

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

COLOR_POS = "#d62728"  # 실제 뇌졸중
COLOR_NEG = "#1f77b4"  # 실제 뇌졸중 아님
COLOR_BEFORE = "#111111"  # 가중치를 맞추기 전 경계선
COLOR_AFTER = "#2ca02c"  # 가중치를 맞춘 뒤 경계선
NAME_COLOR_BEFORE = "검은색"
NAME_COLOR_AFTER = "초록색"


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


def compare_table(before, after):
    """네 칸과 네 지표를 가중치를 맞추기 전·후로 나란히 놓은 표."""
    tp0, fp0, fn0, tn0 = before
    tp1, fp1, fn1, tn1 = after
    m0 = four_metrics(*before)
    m1 = four_metrics(*after)
    rows = {
        "TP 찾아낸 환자": (f"{tp0:,}명", f"{tp1:,}명"),
        "FN 놓친 환자": (f"{fn0:,}명", f"{fn1:,}명"),
        "FP 헛짚은 사람": (f"{fp0:,}명", f"{fp1:,}명"),
        "TN 아님을 맞힌 사람": (f"{tn0:,}명", f"{tn1:,}명"),
    }
    for key in ("정확도", "재현율", "정밀도", "F1"):
        rows[key] = (m0[key], m1[key])
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["맞추기 전", "맞춘 뒤"]
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
st.markdown("<h1 style='margin-bottom:0'>⚖️ 성능 높이기</h1>", unsafe_allow_html=True)
st.caption(
    "뇌졸중을 겪은 사람이 아주 드물어서, 두 모델이 뇌졸중 쪽을 잘 못 맞혀요. "
    "뇌졸중인 사람과 아닌 사람의 가중치를 맞추기 전과 맞춘 뒤를 비교해 봐요. "
    "뇌졸중(stroke=1)을 양성으로 둬요."
)

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결과 주소를 확인해 주세요.\n\n{e}")
    st.stop()

# ---------- 입력 속성 고르기 ('분류 모델' 페이지와 같은 방법) ----------
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

# ---------- 테스트용 / 훈련용 나누기 ('채점표' 페이지와 같은 방법) ----------
# 번호(id) 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정한다.
data = df.sort_values("id").reset_index(drop=True)
is_test = (np.arange(len(data)) % 10) < 3
train_df = data[~is_test]
test_df = data[is_test].reset_index(drop=True)

X_train = train_df[selected].copy()
X_test = test_df[selected].copy()
y_train = train_df["stroke"]
y_test = test_df["stroke"].to_numpy()

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

# ---------- 모델 만들기: 가중치를 맞추기 전 / 맞춘 뒤 ----------
lr_before = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(Xs_train, y_train)
lr_after = LogisticRegression(
    max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
).fit(Xs_train, y_train)

dt_before = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE
).fit(X_train, y_train)
dt_after = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE, class_weight="balanced"
).fit(X_train, y_train)

cells = {
    NAME_LR: (
        count_cells(y_test, lr_before.predict(Xs_test)),
        count_cells(y_test, lr_after.predict(Xs_test)),
    ),
    NAME_DT: (
        count_cells(y_test, dt_before.predict(X_test)),
        count_cells(y_test, dt_after.predict(X_test)),
    ),
}

n_test = len(test_df)
n_test_stroke = int((y_test == 1).sum())
st.write(f"테스트용 {n_test:,}명 가운데 뇌졸중인 사람은 {n_test_stroke:,}명이에요.")

# ---------- 1. 전·후 비교표 ----------
st.header("1. 가중치를 맞추기 전과 맞춘 뒤")
st.caption(
    "가중치를 맞춘다는 것은, 뇌졸중인 사람 전체의 무게와 아닌 사람 전체의 무게가 같아지도록 "
    "한 사람당 무게를 조정하는 거예요. 드문 뇌졸중 쪽을 틀릴 때 더 크게 혼나는 셈이에요. "
    "크기 맞추기와 의사결정트리의 설정(질문 세 번, 마지막 마디 다섯 명 이상, 난수 고정)은 "
    "두 경우 모두 같아요."
)
left, right = st.columns(2)
for col, name in zip((left, right), (NAME_LR, NAME_DT)):
    before, after = cells[name]
    with col:
        st.subheader(name)
        st.table(compare_table(before, after))
        tp0, fp0, fn0, tn0 = before
        tp1, fp1, fn1, tn1 = after
        st.write(
            f"찾아낸 환자 {tp0:,}명에서 {tp1:,}명으로, "
            f"놓친 환자 {fn0:,}명에서 {fn1:,}명으로, "
            f"헛짚은 사람 {fp0:,}명에서 {fp1:,}명으로 바뀌었어요."
        )


def axis_range(col):
    lo, hi = float(X_test[col].min()), float(X_test[col].max())
    span = hi - lo
    if col in BINARY:
        pad = 0.25
    else:
        pad = span * 0.05 if span > 0 else 1.0
    return lo - pad, hi + pad


def jitter(values, feature, rng):
    """0과 1만 있는 속성은 점이 겹치지 않게 살짝 흩어서 찍는다."""
    if feature in BINARY:
        return values + rng.uniform(-0.12, 0.12, len(values)), True
    return values, False


def make_grid(axis_features, axis_values, others, other_medians):
    """축 속성들의 격자에 나머지 속성은 중앙값으로 세워 둔 표를 만든다."""
    grid = pd.DataFrame({f: v for f, v in zip(axis_features, axis_values)})
    for f in others:
        grid[f] = other_medians[f]
    return grid[selected]


def line_note(label, color_name, shape_word, p):
    if p.min() < 0.5 < p.max():
        return f"{label}({color_name})은 그림 안에 있어요."
    if p.max() <= 0.5:
        return (
            f"{label}({color_name})은 이 그림 밖에 있어요. "
            f"그림 안 모든 자리에서 뇌졸중 확률이 0.5보다 낮고, 가장 높은 곳도 {p.max() * 100:.1f}%예요."
        )
    return (
        f"{label}({color_name})은 이 그림 밖에 있어요. "
        f"그림 안 모든 자리에서 뇌졸중 확률이 0.5보다 높고, 가장 낮은 곳도 {p.min() * 100:.1f}%예요."
    )


# ---------- 2. 2차원 산점도 ----------
st.header("2. 산점도와 0.5 경계선 (로지스틱 회귀)")
st.caption(
    "로지스틱 회귀(확률로 답하는 모델)가 뇌졸중 확률 0.5로 가르는 선을 "
    "가중치를 맞추기 전과 맞춘 뒤로 겹쳐 그려요."
)
c1, c2 = st.columns(2)
x_feat = c1.selectbox(
    "가로축", selected, index=0, format_func=lambda f: KOR[f], key="x2d"
)
y_options = [f for f in selected if f != x_feat]
y_feat = c2.selectbox(
    "세로축", y_options, index=0, format_func=lambda f: KOR[f], key="y2d"
)

x0, x1 = axis_range(x_feat)
y0, y1 = axis_range(y_feat)
xs = np.linspace(x0, x1, 160)
ys = np.linspace(y0, y1, 160)
XX, YY = np.meshgrid(xs, ys)

others2 = [f for f in selected if f not in (x_feat, y_feat)]
medians2 = {f: float(X_test[f].median()) for f in others2}
grid2 = make_grid((x_feat, y_feat), (XX.ravel(), YY.ravel()), others2, medians2)
grid2_scaled = scaler.transform(grid2)
p2_before = lr_before.predict_proba(grid2_scaled)[:, 1].reshape(XX.shape)
p2_after = lr_after.predict_proba(grid2_scaled)[:, 1].reshape(XX.shape)

rng = np.random.default_rng(0)
px_vals, jx = jitter(X_test[x_feat].to_numpy(dtype=float), x_feat, rng)
py_vals, jy = jitter(X_test[y_feat].to_numpy(dtype=float), y_feat, rng)

fig2 = go.Figure()
for label, value, color in [
    ("실제로 뇌졸중이 아님", 0, COLOR_NEG),
    ("실제로 뇌졸중", 1, COLOR_POS),
]:
    m = y_test == value
    fig2.add_trace(
        go.Scatter(
            x=px_vals[m],
            y=py_vals[m],
            mode="markers",
            name=label,
            marker=dict(color=color, size=6 if value == 1 else 5, opacity=0.65),
        )
    )

for p, color, name in (
    (p2_before, COLOR_BEFORE, "가중치 맞추기 전 경계선(확률 0.5)"),
    (p2_after, COLOR_AFTER, "가중치 맞춘 뒤 경계선(확률 0.5)"),
):
    if p.min() < 0.5 < p.max():
        fig2.add_trace(
            go.Contour(
                x=xs,
                y=ys,
                z=p,
                contours=dict(start=0.5, end=0.5, size=1, coloring="lines", showlabels=False),
                line=dict(color=color, width=3),
                showscale=False,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=name,
                line=dict(color=color, width=3),
            )
        )

fig2.update_layout(
    height=580,
    xaxis=dict(title=KOR[x_feat], range=[x0, x1]),
    yaxis=dict(title=KOR[y_feat], range=[y0, y1]),
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0),
    margin=dict(t=30),
)
st.plotly_chart(fig2, use_container_width=True)

if others2:
    fixed_text = ", ".join(fmt_value(f, medians2[f]) for f in others2)
    st.write(f"그림에 쓰지 않은 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {fixed_text}")
else:
    st.write("고른 속성이 두 개뿐이라, 세워 두고 계산한 속성은 없어요.")
st.write(line_note("가중치를 맞추기 전의 0.5 경계선", NAME_COLOR_BEFORE, "선", p2_before))
st.write(line_note("가중치를 맞춘 뒤의 0.5 경계선", NAME_COLOR_AFTER, "선", p2_after))

notes2 = []
if jx or jy:
    notes2.append("0과 1만 있는 속성은 점이 겹치지 않게 살짝 흩어서 찍었어요.")
if "bmi" in (x_feat, y_feat):
    notes2.append("비어 있던 체질량지수는 훈련용 중앙값으로 채운 값으로 찍혀 있어요.")
if notes2:
    st.caption(" ".join(notes2))

# ---------- 3. 3차원 산점도 ----------
st.header("3. 3차원 산점도와 0.5 평면 (로지스틱 회귀)")
if len(selected) < 3:
    st.info("입력 속성을 세 개 이상 고르면 3차원 산점도가 나와요.")
else:
    st.caption("마우스로 끌어서 돌려 보고, 휠로 확대·축소할 수 있어요.")
    d1, d2, d3 = st.columns(3)
    x3 = d1.selectbox("x축", selected, index=0, format_func=lambda f: KOR[f], key="x3d")
    y3_options = [f for f in selected if f != x3]
    y3 = d2.selectbox("y축", y3_options, index=0, format_func=lambda f: KOR[f], key="y3d")
    z3_options = [f for f in selected if f not in (x3, y3)]
    z3 = d3.selectbox("z축", z3_options, index=0, format_func=lambda f: KOR[f], key="z3d")

    axes3 = (x3, y3, z3)
    ranges3 = {f: axis_range(f) for f in axes3}
    n_grid = 28
    lin = [np.linspace(*ranges3[f], n_grid) for f in axes3]
    GX, GY, GZ = np.meshgrid(lin[0], lin[1], lin[2], indexing="ij")

    others3 = [f for f in selected if f not in axes3]
    medians3 = {f: float(X_test[f].median()) for f in others3}
    grid3 = make_grid(axes3, (GX.ravel(), GY.ravel(), GZ.ravel()), others3, medians3)
    grid3_scaled = scaler.transform(grid3)
    p3_before = lr_before.predict_proba(grid3_scaled)[:, 1]
    p3_after = lr_after.predict_proba(grid3_scaled)[:, 1]

    rng3 = np.random.default_rng(0)
    pts = []
    jittered3 = False
    for f in axes3:
        v, j = jitter(X_test[f].to_numpy(dtype=float), f, rng3)
        pts.append(v)
        jittered3 = jittered3 or j

    fig3 = go.Figure()
    for label, value, color in [
        ("실제로 뇌졸중이 아님", 0, COLOR_NEG),
        ("실제로 뇌졸중", 1, COLOR_POS),
    ]:
        m = y_test == value
        fig3.add_trace(
            go.Scatter3d(
                x=pts[0][m],
                y=pts[1][m],
                z=pts[2][m],
                mode="markers",
                name=label,
                marker=dict(color=color, size=3 if value == 0 else 4, opacity=0.6),
            )
        )

    for p, color, name in (
        (p3_before, COLOR_BEFORE, "가중치 맞추기 전 0.5 평면"),
        (p3_after, COLOR_AFTER, "가중치 맞춘 뒤 0.5 평면"),
    ):
        if p.min() < 0.5 < p.max():
            fig3.add_trace(
                go.Isosurface(
                    x=GX.ravel(),
                    y=GY.ravel(),
                    z=GZ.ravel(),
                    value=p,
                    isomin=0.499,
                    isomax=0.501,
                    surface_count=1,
                    opacity=0.4,
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    caps=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig3.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=6),
                )
            )

    fig3.update_layout(
        height=680,
        scene=dict(
            xaxis=dict(title=KOR[x3], range=list(ranges3[x3])),
            yaxis=dict(title=KOR[y3], range=list(ranges3[y3])),
            zaxis=dict(title=KOR[z3], range=list(ranges3[z3])),
            aspectmode="cube",
        ),
        legend=dict(orientation="h", yanchor="top", y=0.0, xanchor="left", x=0),
        margin=dict(t=30, l=0, r=0, b=0),
    )
    st.plotly_chart(fig3, use_container_width=True)

    if others3:
        fixed3 = ", ".join(fmt_value(f, medians3[f]) for f in others3)
        st.write(f"그림에 쓰지 않은 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {fixed3}")
    else:
        st.write("세 속성이 모두 축이라, 세워 두고 계산한 속성은 없어요.")
    st.write(line_note("가중치를 맞추기 전의 0.5 평면", NAME_COLOR_BEFORE, "면", p3_before))
    st.write(line_note("가중치를 맞춘 뒤의 0.5 평면", NAME_COLOR_AFTER, "면", p3_after))

    notes3 = []
    if jittered3:
        notes3.append("0과 1만 있는 속성은 점이 겹치지 않게 살짝 흩어서 찍었어요.")
    if "bmi" in axes3:
        notes3.append("비어 있던 체질량지수는 훈련용 중앙값으로 채운 값으로 찍혀 있어요.")
    if notes3:
        st.caption(" ".join(notes3))
