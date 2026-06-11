import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
import requests
import json

# 页面配置（和你之前的保持一致）
st.set_page_config(page_title="中医药知识图谱+AI问答", layout="wide")
st.title("🌿 中医药知识图谱智能系统")

# ---------------------- 1. 数据库连接（原封不动） ----------------------
@st.cache_resource
def init_driver():
    uri = "neo4j+s://cb5cc04e.databases.neo4j.io"
    user = "cb5cc04e"
    pwd = "uzjqMbskUGdIObBV0uRRs6AoTO8gpmetKkHyhd3vuhs"
    return GraphDatabase.driver(uri, auth=(user, pwd))

driver = init_driver()

# ---------------------- 2. 原有实体查询函数（完全不变，只修复None显示问题） ----------------------
def get_entity_info(entity_name):
    with driver.session() as session:
        res = session.run("""
            MATCH (n:Entity {id: $name}) RETURN n
        """, name=entity_name)
        records = list(res)
        if not records:
            return None, None
        
        node = records[0]["n"]
        props = {}
        # 修复：过滤掉None值，避免表格里显示空
        for key, val in dict(node).items():
            props[key] = val if val is not None else ""

        res_rel = session.run("""
            MATCH (n:Entity {id: $name})-[r]-(m)
            RETURN type(r) AS 关系类型, m.id AS 关联实体
        """, name=entity_name)
        rel_list = list(res_rel)
        relations = [rec.data() for rec in rel_list]
    return props, relations

# ---------------------- 3. 原有病症查询函数（原封不动） ----------------------
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

# ---------------------- 4. 新增：大模型调用函数（不影响原有代码） ----------------------
def call_tongyi_api(api_key, graph_context, user_question):
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    prompt = f"""你是专业的中医药顾问，请严格依据下方【知识图谱检索结果】回答用户问题。
如果图谱中无相关信息，请如实告知，不要编造内容。

【知识图谱检索结果】：
{graph_context}

【用户问题】：
{user_question}

请用通俗易懂的中文回答，分点说明更佳：
"""
    payload = {
        "model": "qwen-turbo",
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "parameters": {"temperature": 0.3, "max_tokens": 2000}
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    return response.json()["output"]["text"]

# ---------------------- 5. 新增：图谱上下文检索（给AI用，不影响原有查询） ----------------------
def search_graph_context(question):
    context = []
    with driver.session() as session:
        entity_res = session.run("""
            MATCH (n:Entity) WHERE n.id CONTAINS $kw RETURN n.id LIMIT 10
        """, kw=question)
        entity_list = [rec["n.id"] for rec in list(entity_res)]
        if entity_list:
            context.append(f"匹配实体：{', '.join(entity_list)}")
        for entity in entity_list[:3]:
            # 同时获取属性和关系，给AI更完整的信息
            node_res = session.run("""
                MATCH (n:Entity {id:$e}) RETURN n
            """, e=entity)
            node = list(node_res)[0]["n"]
            props_str = ", ".join([f"{k}:{v}" for k, v in dict(node).items() if v])
            context.append(f"{entity} 属性：{props_str}")

            rel_res = session.run("""
                MATCH (n:Entity{id:$e})-[r]-(m)
                RETURN n.id, type(r), m.id LIMIT 10
            """, e=entity)
            for rec in list(rel_res):
                s, r, t = rec.values()
                context.append(f"{s} —{r}→ {t}")
    return "\n".join(context) if context else "知识图谱中未查询到相关信息"

# ---------------------- 6. 页面菜单（在原有基础上加AI问答） ----------------------
menu = st.sidebar.selectbox(
    "功能菜单",
    ["实体查询", "病症找药", "🤖 AI智能问答"]
)

# 原有：实体查询页面（完全不变，只修复None显示）
if menu == "实体查询":
    st.subheader("📌 药材/实体查询")
    entity_name = st.text_input("输入实体名称（如：丁香）", "丁香")
    if st.button("查询"):
        props, relations = get_entity_info(entity_name)
        if props is None:
            st.warning("未找到该实体，请检查名称是否正确")
        else:
            st.markdown("### 基本属性")
            st.dataframe(pd.DataFrame(list(props.items()), columns=["属性", "值"]), use_container_width=True)
            if relations:
                st.markdown("### 关联关系")
                st.dataframe(pd.DataFrame(relations), use_container_width=True)
            else:
                st.info("该实体暂无关联关系")

# 原有：病症找药页面（完全不变）
elif menu == "病症找药":
    st.subheader("💊 根据病症查询推荐药材")
    disease = st.text_input("输入病症名称（如：肾虚阳痿）", "肾虚阳痿")
    if st.button("查询"):
        df = query_herbs_for_disease(disease)
        if df.empty:
            st.warning("未找到相关药材")
        else:
            st.success(f"找到{len(df)}种相关药材")
            st.dataframe(df, use_container_width=True)

# 新增：AI智能问答页面（不影响原有功能）
elif menu == "🤖 AI智能问答":
    st.subheader("🤖 中医药AI问答")
    st.info("结合知识图谱回答药材、病症相关问题，不编造内容")
    user_question = st.text_area("请输入问题", placeholder="例如：丁香的性味归经是什么？可以治疗哪些病症？", height=100)
    if st.button("开始问答", type="primary") and user_question.strip():
        with st.spinner("正在检索图谱并思考..."):
            api_key = st.secrets["DASHSCOPE_API_KEY"]
            graph_ctx = search_graph_context(user_question)
            with st.expander("📚 知识图谱检索详情", expanded=False):
                st.write(graph_ctx)
            try:
                answer = call_tongyi_api(api_key, graph_ctx, user_question)
                st.markdown("### ✅ AI 回答")
                st.write(answer)
            except Exception as e:
                st.error(f"调用大模型失败：{str(e)}")
