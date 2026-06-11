import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
import requests
import time
import hashlib
import base64

# 页面基础配置
st.set_page_config(page_title="中医药知识图谱+AI问答", layout="wide")
st.title("🌿 中医药知识图谱智能系统")

# ===================== Neo4j 数据库连接 =====================
@st.cache_resource
def init_driver():
    uri = st.secrets["neo4j_uri"]
    user = st.secrets["neo4j_user"]
    pwd = st.secrets["neo4j_password"]
    return GraphDatabase.driver(uri, auth=(user, pwd))

driver = init_driver()

# ===================== 实体查询函数 =====================
def get_entity_info(entity_name):
    with driver.session() as session:
        res = session.run("MATCH (n:Entity {id: $name}) RETURN n", name=entity_name)
        records = list(res)
        if not records:
            return None, None
        
        node = records[0]["n"]
        props = {}
        for key, val in dict(node).items():
            if val is not None and val != "":
                props[key] = val

        res_rel = session.run("""
            MATCH (n:Entity {id: $name})-[r]-(m)
            RETURN type(r) AS 关系类型, m.id AS 关联实体
        """, name=entity_name)
        rel_list = list(res_rel)
        relations = [rec.data() for rec in rel_list]
    return props, relations

# ===================== 病症查药材函数 =====================
def query_herbs_for_disease(disease_name):
    with driver.session() as session:
        res = session.run("""
            MATCH (m:Entity)-[r]->(d:Entity {id: $disease})
            WHERE type(r) = '治疗'
            RETURN DISTINCT m.id AS 推荐药材
        """, disease=disease_name)
        records = list(res)
        data = [rec.data() for rec in records]
        return pd.DataFrame(data)

# ===================== 精简版图谱检索 =====================
def search_graph_context(question):
    context = []
    stop_words = ["用什么", "什么药", "检测", "治疗", "含有", "属于", "？", "，", "。"]
    temp_q = question
    for word in stop_words:
        temp_q = temp_q.replace(word, "")

    keywords = []
    for token in temp_q.split():
        if len(token) > 1:
            keywords.append(token)
    keywords.append(question)

    entity_ids = set()
    with driver.session() as session:
        for kw in keywords:
            res = session.run("""
                MATCH (n:Entity)
                WHERE n.id = $kw OR n.id CONTAINS $kw
                RETURN n.id LIMIT 3
            """, kw=kw)
            for rec in res:
                entity_ids.add(rec["n.id"])

        entity_ids = list(entity_ids)[:2]
        if entity_ids:
            context.append(f"匹配实体：{', '.join(entity_ids)}")
            for entity_id in entity_ids:
                res_node = session.run("MATCH (n:Entity {id: $id}) RETURN n", id=entity_id)
                node = list(res_node)[0]["n"]
                props_str = ", ".join([f"{k}:{v}" for k, v in dict(node).items() if v])
                if props_str:
                    context.append(f"{entity_id} 属性：{props_str}")

                res_rel = session.run("""
                    MATCH (a:Entity)-[r]-(b:Entity)
                    WHERE a.id = $id OR b.id = $id
                    RETURN a.id, type(r), b.id LIMIT 5
                """, id=entity_id)
                for rec in res_rel:
                    s, r, t = rec.values()
                    context.append(f"{s} - {r} - {t}")

    return "\n".join(context) if context else "暂无相关图谱数据"

# ===================== 讯飞星火 API（适配你提供的密钥和地址） =====================
def call_spark(appid, api_key, api_secret, graph_context, user_question):
    url = "https://spark-api.xf-yun.com/v1.1/chat"
    host = "spark-api.xf-yun.com"
    path = "/v1.1/chat"
    timestamp = str(int(time.time()))

    # 按讯飞官方规则生成签名
    hmac_data = api_key + timestamp
    md5 = hashlib.md5()
    md5.update(hmac_data.encode("utf-8"))
    checksum = md5.hexdigest()
    auth = base64.b64encode(f"{appid}:{api_secret}".encode()).decode()

    headers = {
        "Authorization": f"Bearer {auth},{timestamp},{checksum}",
        "Content-Type": "application/json",
        "Host": host
    }

    # 精简提示词，减少超时
    system_prompt = "依据下方图谱数据回答，无数据请建议咨询中医师，禁止编造内容。"
    full_text = f"图谱数据：{graph_context}\n用户问题：{user_question}"

    payload = {
        "header": {"app_id": appid},
        "parameter": {"chat": {"domain": "lite", "temperature": 0.3}},
        "payload": {
            "message": {
                "text": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_text}
                ]
            }
        }
    }

    # 增加1次重试，超时15秒
    for _ in range(2):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            return response.json()["payload"]["choices"]["text"][0]["content"]
        except:
            time.sleep(1)
            continue
    return "接口请求超时/网络异常，请稍后重试"

# ===================== 页面交互 =====================
menu = st.sidebar.selectbox(
    "功能菜单",
    ["实体查询", "病症找药", "🤖 AI智能问答"]
)

# 实体查询
if menu == "实体查询":
    st.subheader("📌 药材/实体查询")
    entity_name = st.text_input("输入实体名称（如：丁香）", "丁香")
    if st.button("查询"):
        props, relations = get_entity_info(entity_name)
        if props is None:
            st.warning("未找到该实体，请检查名称是否正确")
        else:
            st.markdown("### 基本属性")
            st.dataframe(pd.DataFrame(list(props.items()), columns=["属性", "值"]), use_container_width)
            if relations:
                st.markdown("### 关联关系")
                st.dataframe(pd.DataFrame(relations), use_container_width)
            else:
                st.info("该实体暂无关联关系")

# 病症找药
elif menu == "病症找药":
    st.subheader("💊 根据病症查询推荐药材")
    disease = st.text_input("输入病症名称（如：肾虚阳痿）", "肾虚阳痿")
    if st.button("查询"):
        df = query_herbs_for_disease(disease)
        if df.empty:
            st.warning("未找到相关药材")
        else:
            st.success(f"找到{len(df)}种相关药材")
            st.dataframe(df, use_container_width)

# AI 智能问答
elif menu == "🤖 AI智能问答":
    st.subheader("🤖 AI智能问答")
    st.info("基于知识图谱作答，仅使用图谱内数据")
    user_question = st.text_area("请输入问题", placeholder="例如：肾虚腰痛用什么药？", height=100)

    if st.button("开始问答", type="primary") and user_question.strip():
        with st.spinner("正在检索并请求AI..."):
            # 读取密钥并捕获异常
            try:
                appid = st.secrets["XF_APPID"]
                api_key = st.secrets["XF_API_KEY"]
                api_secret = st.secrets["XF_API_SECRET"]
            except KeyError as e:
                st.error(f"密钥配置缺失！请检查Secrets中是否存在 {e}")
                st.stop()

            graph_ctx = search_graph_context(user_question)
            with st.expander("📚 知识图谱检索详情", expanded=False):
                st.write(graph_ctx)

            # 调用AI
            answer = call_spark(appid, api_key, api_secret, graph_ctx, user_question)
            st.markdown("### ✅ AI 回答")
            st.write(answer)
