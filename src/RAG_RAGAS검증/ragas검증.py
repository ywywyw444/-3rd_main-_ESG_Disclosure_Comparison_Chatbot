import pandas as pd
from datetime import datetime
import csv
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import re
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMListwiseRerank

import key
openai_api_key = key.key['OPENAI_API_KEY']

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, openai_api_key=openai_api_key)
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

CSV_PATH = "question1.csv"
def init_csv():
    if not os.path.isfile(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "question", "standard",
                "context_recall", "context_precision",
                "faithfulness", "answer_relevancy"
            ])

def clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\d{1,2}$", stripped):
            continue
        cleaned.append(stripped)
    return ' '.join(cleaned)


vectorstore_paths = {
    "IFRS_S1": r"C:\project2\esg_chatbot_project\chat_bot\vector_finish\KOR\vectorstores\KOR_IFRS_S1_2nd",
    "IFRS_S2": r"C:\project2\esg_chatbot_project\chat_bot\vector_finish\KOR\vectorstores\KOR_IFRS_S2_2nd",
    "KSSB1": r"C:\project2\esg_chatbot_project\chat_bot\vector_finish\KOR\vectorstores\KSSB_01_2nd",
    "KSSB2": r"C:\project2\esg_chatbot_project\chat_bot\vector_finish\KOR\vectorstores\KSSB_02_2nd",
    "KSSB101": r"C:\project2\esg_chatbot_project\chat_bot\vector_finish\KOR\vectorstores\KSSB_101_2nd"
}
dbs = {
    name: FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
    for name, path in vectorstore_paths.items()
}

# def translate_to_english(korean_query: str) -> str:
#     messages = [
#         SystemMessage(content="You are a professional English translator. Translate the following Korean question into natural and accurate English."),
#         HumanMessage(content=korean_query)
#     ]
#     return llm.invoke(messages).content.strip()


def classify_relevant_standards(query: str):
    prompt = f"""
사용자 질문: "{query}"

이 질문과 가장 관련 있는 ESG 기준서를 아래 중에서 선택해줘. 복수 선택 가능해.
가능한 선택: GRI, IFRS_S1, IFRS_S2, TCFD, KSSB1, KSSB2, KSSB101
형식: [\"GRI\", \"IFRS_S2\"] 처럼 Python 리스트로만 답해줘.
"""
    response = llm.invoke([
        SystemMessage(content="ESG 기준서 라우팅 전문가"),
        HumanMessage(content=prompt)
    ])
    try:
        return eval(response.content)
    except:
        return list(dbs.keys())

def process_standard(standard, db, q_ko):
    ret = db.as_retriever(search_kwargs={"k":4})
    mq = MultiQueryRetriever.from_llm(retriever=ret, llm=llm)
    rr = LLMListwiseRerank.from_llm(llm=llm, top_n=2)
    comp = ContextualCompressionRetriever(base_retriever=mq, base_compressor=rr)
    docs = comp.invoke(q_ko)
    if not docs:
        return None

    context = "\n\n".join([d.page_content for d in docs])
    answer = llm.invoke([SystemMessage(content=f"Answer accurately from docs:\n{context}"), HumanMessage(content=q_ko)]).content.strip()

    ans_emb = embedding_model.embed_query(answer)
    chunk_embs = [embedding_model.embed_query(d.page_content) for d in docs]
    sims = [np.dot(ans_emb, ce)/(np.linalg.norm(ans_emb)*np.linalg.norm(ce)) for ce in chunk_embs]
    thr = 0.5
    y_pred = [1 if s >= thr else 0 for s in sims]

    context_recall = sum(y_pred) / len(chunk_embs)
    context_precision = sum(y_pred) / len(y_pred)
    ctx_emb = embedding_model.embed_query(context)
    faithfulness = np.dot(ans_emb, ctx_emb) / (np.linalg.norm(ans_emb)*np.linalg.norm(ctx_emb))
    q_emb = embedding_model.embed_query(q_ko)
    answer_relevancy = np.dot(q_emb, ans_emb) / (np.linalg.norm(q_emb)*np.linalg.norm(ans_emb))

    return {
        "standard": standard,
        "answer": answer,
        "metrics": {
            "context_recall": context_recall,
            "context_precision": context_precision,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy
        }
    }

# ✅ 전체 실행
def run_all_questions_from_csv(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8')
    init_csv()

    for idx, row in df.iterrows():
        q = row["query"]
        # q_en = translate_to_english(q)
        standards = classify_relevant_standards(q)

        all_metrics = []
        for std in standards:
            if std not in dbs:
                continue
            result = process_standard(std, dbs[std], q)
            if result:
                all_metrics.append(result["metrics"])

        if not all_metrics:
            print(f"❌ {idx+1}/{len(df)} 실패 (관련 기준서 없음): {q}")
            continue

        # ✅ 평균 계산
        avg_metrics = {
            "context_recall": np.mean([m["context_recall"] for m in all_metrics]),
            "context_precision": np.mean([m["context_precision"] for m in all_metrics]),
            "faithfulness": np.mean([m["faithfulness"] for m in all_metrics]),
            "answer_relevancy": np.mean([m["answer_relevancy"] for m in all_metrics])
        }

        # ✅ CSV 기록
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(), q, "AVG",
                f"{avg_metrics['context_recall']:.4f}",
                f"{avg_metrics['context_precision']:.4f}",
                f"{avg_metrics['faithfulness']:.4f}",
                f"{avg_metrics['answer_relevancy']:.4f}"
            ])
        print(f"✅ {idx+1}/{len(df)} 질문 완료: {q}")

# 실행
run_all_questions_from_csv("question1.csv")

