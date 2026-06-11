import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
import requests
import time
import hashlib
import base64
from requests.adapters import HTTPAdapter
from urllib3.util.retry

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

# ===================== 极简图谱检索（最小化传输内容） =====================
def search_graph_context(question):
    context = []
    # 过滤停用词
    stop_words = ["用什么", "什么药", "检测", "治疗", "含有", "属于", "？", "，", "。"]
    temp_q = question
    for word in stop_words:
        temp_q = temp_q.replace(word, "")

    keywords = [w for w in temp_q.split() if len(w) > 1]
    keywords.append(question)

    entity_ids = set()
    with driver.session() as session:
        # 限制实体数量
        for kw in keywords:
            res = session.run("MATCH (n:Entity) WHERE n.id CONTAINS $kw RETURN n.id LIMIT 2", kw=kw)
            for rec in res:
                entity_ids.add(rec["n.id"])
        entity_ids = list(entity_ids)[:2]

        if entity_ids:
            context.append(f"实体：{', '.join(entity_ids)}")
            for e_id in entity_ids:
                # 实体属性
                node = session.run("MATCH (n{id:$id}) RETURN n", id=e_id).single()["n"]
                props = [f"{k}:{v}" for k, v in dict(node).items() if v]
                if props:
                    context.append(f"{e_id}：{' | '.join(props)}")
                # 限制关系数量
                rels = session.run("""
                MATCH (a)-[r]-(b) WHERE a.id=$id RETURN a,type(r),b LIMIT 4
                """, id=e_id)
                for r in rels:
                    context.append(f"{r['a'].id} - {r[1]} - {r['b'].id}")
    return "\n".join(context) if context else "无图谱数据"

# ===================== 讯飞星火 API（网络深度优化） =====================
def create_retry_session():
    """创建带自动重试的请求会话，解决网络波动"""
    session = requests.Session()
    # 重试规则：超时/连接错误自动重试3次，间隔1s
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def call_spark(appid, api_key, api_secret, graph_ctx, user_q):
    url = "https://spark-api.xf-yun.com/v1.1/chat"
    timestamp = str(int(time.time()))
    # 签名计算
    md5 = hashlib.md5((api_key + timestamp).encode("utf-8"))
    checksum = md5.hexdigest()
    auth = base64.b64encode(f"{appid}:{api_secret}".encode()).decode()
    headers = {
        "Authorization": f"Bearer {auth},{timestamp},{checksum}",
        "Content-Type": "application/json"
    }
    # 极简提示词
    sys_prompt = "仅依据下方图谱内容回答，无数据请建议咨询中医师，禁止编造。"
    full_msg = f"图谱：{graph_ctx}\n问题：{user_q}"
    payload = {
        "header": {"app_id": appid},
        "parameter": {"chat": {"domain": "lite", "temperature": 0.3}},
        "payload": {"message": {"text": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": full_msg}
        ]}}
    }
    session = create_retry_session()
    # 拆分超时：连接5秒，读取30秒（适配大模型响应）
    try:
        response = session.post(url, headers=headers, json=payload, timeout=(5, 30))
        response.raise_for_status()
        return response.json()["payload"]["choices"]["text"][0]["content"]
    except requests.exceptions.ConnectionError:
        return "错误：无法连接讯飞接口，网络不通"
    except requests.exceptions.ReadTimeout:
        return "错误：接口读取超时，请稍后重试"
    except requests.exceptions.Timeout:
        return "错误：请求超时，请稍后重试"
    except Exception as e:
        return f"接口异常：{str(e)}"

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
            st.warning("未找到该实体，请检查名称")
        else:
            st.dataframe(pd.DataFrame(list(props.items()), columns=["属性", "值"]), use_container_width)
            if relations:
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
    st.info("基于知识图谱作答")
    user_question = st.text_area("请输入问题", placeholder="例如：丁香的功效是什么？", height=100)

    if st.button("开始问答", type="primary") and user_question.strip():
        with st.spinner("正在处理..."):
            # 读取密钥
            try:
                appid = st.secrets["XF_APPID"]
                api_key = st.secrets["XF_API_KEY"]
                api_secret = st.secrets["XF_API_SECRET"]
            except KeyError:
                st.error("密钥配置缺失！")
                st.stop()
            # 检索图谱
            graph_data = search_graph_context(user_question)
            with st.expander("📚 知识图谱检索详情", expanded=False):
                st.write(graph_data)
            # 调用AI
            res = call_spark(appid, api_key, api_secret, graph_data, user_question)
            st.markdown("### ✅ AI 回答")
            st.write(res)
