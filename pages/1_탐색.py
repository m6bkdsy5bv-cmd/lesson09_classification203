import pandas as pd
import plotly.express as px
import streamlit as st

TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

st.set_page_config(page_title=f"탐색 · {TITLE}", page_icon="🔍", layout="wide")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.markdown("<h1 style='margin-bottom:0'>🔍 탐색</h1>", unsafe_allow_html=True)
st.caption("데이터를 그림과 표로 살펴봐요.")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결과 주소를 확인해 주세요.\n\n{e}")
    st.stop()

df = df.copy()
df["뇌졸중 여부"] = df["stroke"].map({0: "겪지 않음", 1: "겪음"})
group_order = {"뇌졸중 여부": ["겪지 않음", "겪음"]}

# ---------- 1. 분포 히스토그램 ----------
st.header("1. 나이와 평균 혈당의 분포")
left, right = st.columns(2)
with left:
    fig = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "사람 수"},
    )
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=40,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

# ---------- 2. 뇌졸중 여부별 상자그림 ----------
st.header("2. 뇌졸중을 겪은 사람과 겪지 않은 사람 비교")
left, right = st.columns(2)
with left:
    fig = px.box(
        df,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        category_orders=group_order,
        title="나이",
        labels={"age": "나이"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.box(
        df,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        category_orders=group_order,
        title="평균 혈당",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("두 그룹의 평균값")
mean_table = (
    df.groupby("뇌졸중 여부")
    .agg(
        사람_수=("stroke", "size"),
        나이_평균=("age", "mean"),
        평균혈당_평균=("avg_glucose_level", "mean"),
    )
    .reindex(["겪지 않음", "겪음"])
    .reset_index()
    .rename(
        columns={
            "뇌졸중 여부": "구분",
            "사람_수": "사람 수",
            "나이_평균": "나이 평균",
            "평균혈당_평균": "평균 혈당 평균",
        }
    )
)
mean_table["나이 평균"] = mean_table["나이 평균"].round(2)
mean_table["평균 혈당 평균"] = mean_table["평균 혈당 평균"].round(2)
st.dataframe(mean_table, use_container_width=True, hide_index=True)

# ---------- 3. 고혈압 / 심장병별 뇌졸중 비율 ----------
st.header("3. 고혈압·심장병에 따른 뇌졸중 비율")


def rate_chart(column, name):
    rate = (
        df.groupby(column)["stroke"]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": "비율", "size": "사람 수"})
    )
    rate["비율"] = (rate["비율"] * 100).round(2)
    rate[name] = rate[column].map({0: "없음", 1: "있음"})
    fig = px.bar(
        rate,
        x=name,
        y="비율",
        color=name,
        text="비율",
        category_orders={name: ["없음", "있음"]},
        hover_data={"사람 수": True},
        title=f"{name}에 따른 뇌졸중 비율",
        labels={"비율": "뇌졸중 비율(%)"},
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="뇌졸중 비율(%)")
    fig.update_yaxes(range=[0, max(rate["비율"].max() * 1.25, 1)])
    return fig


left, right = st.columns(2)
with left:
    st.plotly_chart(rate_chart("hypertension", "고혈압"), use_container_width=True)
with right:
    st.plotly_chart(rate_chart("heart_disease", "심장병"), use_container_width=True)

# ---------- 4. bmi가 빈 사람들 ----------
st.header("4. 체질량지수(bmi)가 비어 있는 사람들")
missing = df[df["bmi"].isna()]
n_missing = len(missing)
n_missing_stroke = int(missing["stroke"].sum())
n_total = len(df)
n_total_stroke = int(df["stroke"].sum())

bmi_table = pd.DataFrame(
    {
        "구분": ["bmi가 빈 사람", "전체"],
        "사람 수": [n_missing, n_total],
        "뇌졸중 사람 수": [n_missing_stroke, n_total_stroke],
        "뇌졸중 비율(%)": [
            round(n_missing_stroke / n_missing * 100, 2) if n_missing else None,
            round(n_total_stroke / n_total * 100, 2) if n_total else None,
        ],
    }
)
st.dataframe(bmi_table, use_container_width=True, hide_index=True)

# ---------- 5. 흡연 상태별 사람 수 ----------
st.header("5. 흡연 상태별 사람 수")
smoking = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
smoking["흡연 상태"] = smoking["흡연 상태"].fillna("(빈 값)")
st.dataframe(smoking, use_container_width=True, hide_index=True)
