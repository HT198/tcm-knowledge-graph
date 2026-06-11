import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
import requests

# 页面基础配置
st.set_page_config(page_title="中医药知识图谱+AI问答", layout="wide")
st.title("🌿 中医药知识图谱智能系统")

# ===================== Neo4j 数据库连接 =====================
@st.cache_resource
def init_driver():
    uri = "neo4j+s://cb5cc04e.databases.neo4j.io"
    user = "cb5cc04e"
    pwd = "uzjqMbskUGdIObBV0uRRs6AoTO8gpmetKkHyhd3vuhs"
    return GraphDatabase.driver(uri, auth=(user, pwd))

driver = init_driver()

# ===================== 原有功能函数 =====================
def get_entity_info(entity_name):
    """查询实体属性+关联关系"""
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

def query_herbs_for_disease(disease_name):
    """根据病症查询药材"""
    with driver.session() as session:
        res = session.run("""
            MATCH (m:Entity)-[r]->(d:Entity {id: $disease})
            WHERE type(r) = '治疗'
            RETURN DISTINCT m.id AS 推荐药材
        """, disease=disease_name)
        records = list(res)
        data = [rec.data() for rec in records]
        return pd.DataFrame(data)

# ===================== 通用图谱检索函数 =====================
def search_graph_context(question):
    context = []
    # 过滤无用助词
    stop_words = ["用什么", "什么药", "检测", "治疗", "含有", "属于", "？", "，", "。"]
    temp_q = question
    for word in stop_words:
        temp_q = temp_q.replace(word, "")

    # 拆分关键词
    keywords = []
    for token in temp_q.split():
        if len(token) > 1:
            keywords.append(token)
    keywords.append(question)

    entity_ids = set()
    with driver.session() as session:
        # 1. 关键词匹配实体
        for kw in keywords:
            res = session.run("""
                MATCH (n:Entity)
                WHERE n.id = $kw OR n.id CONTAINS $kw
                RETURN n.id
            """, kw=kw)
            for rec in res:
                entity_ids.add(rec["n.id"])

        entity_ids = list(entity_ids)
        if entity_ids:
            context.append(f"✅ 匹配到的实体：{', '.join(entity_ids)}")

            # 2. 遍历实体：属性 + 所有关系
            for entity_id in entity_ids[:3]:
                res_node = session.run("MATCH (n:Entity {id: $id}) RETURN n", id=entity_id)
                node = list(res_node)[0]["n"]
                props_str = ", ".join([f"{k}:{v}" for k, v in dict(node).items() if v])
                if props_str:
                    context.append(f"📋 {entity_id} 属性：{props_str}")

                res_rel = session.run("""
                    MATCH (a:Entity)-[r]-(b:Entity)
                    WHERE a.id = $id OR b.id
                    RETURN a.id, type(r), b.id
                """, id=entity_id)
                for rec in res_rel:
                    s, r, t = rec.values()
                    context.append(f"🔗 {s} —[{r}]→ {t}")

        # 3. 兜底：全局关系匹配
        for kw in keywords:
            res = session.run("""
                MATCH (s:Entity)-[r]->(t:Entity)
                WHERE s.id CONTAINS $kw OR t.id CONTAINS $kw
                RETURN s.id, type(r), t.id
            """, kw=kw)
            for rec in res:
                s, r, t = rec.values()
                context.append(f"🔍 {s} —[{r}]→ {t}")

    return "\n".join(context) if context else "知识图谱中未查询到相关信息"

# ===================== 智谱 GLM-4-Flash 接口（修复鉴权） =====================
def call_zhipu_api(app_id, app_secret, graph_context, user_question):
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {"Content-Type": "application/json"}

    system_prompt = """你是专业中医药顾问，严格遵守规则：
1. 只依据【知识图谱检索结果】回答问题；
2. 图谱有内容就如实总结，绝对不要编造图谱以外的药材、方剂、知识；
3. 图谱无相关信息，请直接说明，并建议咨询专业中医师；
4. 回答通俗易懂，条理清晰。"""

    user_content = f"""【知识图谱检索结果】
{graph_context}

【用户问题】
{user_question}"""

    payload = {
        "model": "glm-4-flash",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content"}
        ],
        "temperature": 0.3
    }

    # 正确传参：auth=(app_id, app_secret)
    response = requests.post(url, headers=headers, json=payload, auth=(app_id, app_secret))
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ===================== 页面菜单与交互 =====================
menu = st.sidebar.selectbox(
    "功能菜单",
    ["实体查询", "病症找药", "🤖 AI智能问答"]
)

# 1. 实体查询
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

# 2. 病症找药
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

# 3. AI 智能问答
elif menu == "🤖 AI智能问答":
    st.subheader("🤖 AI智能问答")
    st.info("基于知识图谱作答，仅使用图谱内数据")
    user_question = st.text_area("请输入问题", placeholder="例如：肾虚腰痛用什么药？", height=100)

    if st.button("开始问答", type="primary") and user_question.strip():
        with st.spinner("正在检索图谱并思考..."):
            # 从Secrets读取拆分后的密钥
            app_id = st.secrets["ZHIPU_APP_ID"]
            app_secret = st.secrets["ZHIPU_APP_SECRET"]
            
            graph_ctx = search_graph_context(user_question)
            with st.expander("📚 知识图谱检索详情", expanded=False):
                st.write(graph_ctx)
            
            try:
                answer = call_zhipu_api(app_id, app_secret, graph_ctx, user_question)
                st.markdown("### ✅ AI 回答")
                st.write(answer)
            except Exception as e:
                st.error(f"调用大模型失败：{str(e)}")
