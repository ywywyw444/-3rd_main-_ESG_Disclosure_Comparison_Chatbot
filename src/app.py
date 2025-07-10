import streamlit as st
from global_chatbot import run_eng_chatbot_app
from korean_chatbot import run_kor_chatbot_app

# 페이지 설정
st.set_page_config(page_title="ESG 공시 챗봇", layout="wide")
st.title("🌱 ESG 공시 기준 플랫폼")

# 챗봇 선택 라디오
st.sidebar.header("🤖 챗봇 선택")
bot_type = st.sidebar.radio(
    "사용할 챗봇을 선택하세요",
    ("🌐 글로벌 공시 기준 챗봇", "🌺 국문 기준 챗봇 (KSSB/IFRS)")
)

# 선택된 챗봇 실행
if bot_type == "🌐 글로벌 공시 기준 챗봇":
    st.subheader(" GRI / IFRS S1/S2 / TCFD 기준서 비교")
    run_eng_chatbot_app()
else:
    st.subheader(" KSSB / IFRS 국문 기준서 비교")
    run_kor_chatbot_app()
