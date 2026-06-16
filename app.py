import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
import requests
import json

# ===================== 全局配置（密钥/地址，无需修改） =====================
neo4j_uri = "neo4j+s://cb5cc04e.databases.neo4j.io"
neo4j_user = "cb5cc04e"
neo4j_password = "uzjqMbskUGdIObBV0uRRs6AoTO8gpmetKkHyhd3vuhs"
DASHSCOPE_API_KEY = "sk-27a3eb03063b4a61acea9abb4f06eed8"

# 页面基础配置
st.set_page_config(page_title="中医药知识图谱+AI问答", layout="wide")
st.title("🌿 中医药知识图谱智能系统")

# ===================== 数据库会话函数 =====================
def get_db_session():
    """获取Neo4j会话，适配Streamlit Cloud"""
    try:
        driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password),
            database="neo4j",
            connection_timeout=20,
            max_connection_lifetime=300
        )
        return driver.session()
    except Exception as e:
        st.error(f"数据库连接失败：{str(e)}")
        return None

# ===================== 数据库查询函数 =====================
def get_entity_info(entity_name):
    """查询实体属性+关联关系"""
    session = get_db_session()
    if session is None:
        return {}, []
    try:
        entity_name = entity_name.strip()
        # 查询实体属性
        res = session.run("""
            MATCH (n:Entity {id: $name}) RETURN n
        """, name=entity_name)
        records = list(res)
        if not records:
            return {}, []

        node = records[0]["n"]
        props = {}
        for key, val in dict(node).items():
            if val is not None and val != "":
                props[key] = val

        # 查询关联关系
        res_rel = session.run("""
            MATCH (n:Entity {id: $name})-[r]-(m)
            RETURN type(r) AS 关系类型, m.id AS 关联实体
        """, name=entity_name)
        rel_list = list(res_rel)
        relations = [rec.data() for rec in rel_list]
        return props, relations
    finally:
        session.close()

def query_herbs_for_disease(disease_name):
    """根据病症查询对应药材（独立页面使用）"""
    session = get_db_session()
    if session is None:
        return pd.DataFrame()
    try:
        disease_name = disease_name.strip()
        res = session.run("""
            MATCH (m:Entity)-[r]->(d:Entity {id: $disease})
            WHERE type(r) = '治疗'
            RETURN DISTINCT m.id AS 推荐药材
        """, disease=disease_name)
        records = list(res)
        data = [rec.data() for rec in records]
        return pd.DataFrame(data)
    finally:
        session.close()

def query_disease_by_herb(herb_name):
    """根据药材查询可治疗的病症（实体页专用）"""
    session = get_db_session()
    if session is None:
        return pd.DataFrame()
    try:
        herb_name = herb_name.strip()
        res = session.run("""
            MATCH (d:Entity {id: $herb})-[r]->(dis:Entity)
            WHERE type(r) = '治疗'
            RETURN DISTINCT dis.id AS 对应病症
        """, herb=herb_name)
        records = list(res)
        data = [rec.data() for rec in records]
        return pd.DataFrame(data)
    finally:
        session.close()

def fuzzy_search_all(keyword):
    """全局模糊查询"""
    session = get_db_session()
    if session is None:
        return pd.DataFrame()
    try:
        keyword = keyword.strip()
        cypher = """
        MATCH (n:Entity)
        WHERE n.id CONTAINS $kw
        RETURN DISTINCT n.id AS 实体名称 LIMIT 50
        """
        res = session.run(cypher, kw=keyword)
        records = list(res)
        data = [rec.data() for rec in records]
        return pd.DataFrame(data)
    finally:
        session.close()

# ===================== 大模型调用 & 图谱上下文 =====================
def call_tongyi_api(api_key, graph_context, user_question):
    """调用阿里云通义千问"""
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    prompt = f"""你是专业的中医药顾问，必须严格遵循以下规则回答问题：
1. 优先依据【知识图谱检索结果】中的信息作答。
2. 如果图谱中有明确的【治疗】关系（如“八角茴香 → 治疗 → 肾虚腰痛”），必须以该药材为核心回答。
3. 禁止编造图谱中没有的药材、方剂。
4. 图谱中提到的药材，可以结合它的属性补充说明。
5. 如果图谱无数据，可说明情况并建议咨询专业中医师。

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

def search_graph_context(question):
    """从图谱检索上下文给大模型"""
    stop_words = ["用什么", "什么药", "检测", "治疗", "含有", "属于"]
    for word in stop_words:
        question = question.replace(word, "")
    context = []
    session = get_db_session()
    if session is None:
        return "数据库连接异常，无法查询图谱"
    try:
        keywords = []
        for token in question.replace("？", "").split():
            if len(token) > 1:
                keywords.append(token)
        keywords.append(question)

        entity_ids = set()
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
            for entity_id in entity_ids[:3]:
                res = session.run("""
                    MATCH (n:Entity {id: $id}) RETURN n
                """, id=entity_id)
                node = list(res)[0]["n"]
                props_str = ", ".join([f"{k}:{v}" for k, v in dict(node).items() if v])
                if props_str:
                    context.append(f"📋 {entity_id} 属性：{props_str}")

                res_rel = session.run("""
                    MATCH (a:Entity)-[r]-(b:Entity)
                    WHERE a.id = $id OR b.id = $id
                    RETURN a.id, type(r), b.id
                """, id=entity_id)
                for rec in res_rel:
                    s, r, t = rec.values()
                    context.append(f"🔗 图谱关系：{s} —[{r}]→ {t}")

        for kw in keywords:
            res = session.run("""
                MATCH (s:Entity)-[r]->(t:Entity)
                WHERE s.id CONTAINS $kw OR t.id CONTAINS $kw
                RETURN s.id, type(r), t.id
            """, kw=kw)
            for rec in res:
                s, r, t = rec.values()
                context.append(f"🔍 关联关系：{s} —[{r}]→ {t}")
    finally:
        session.close()
    return "\n".join(context) if context else "知识图谱中未查询到相关信息"

# ===================== 侧边栏菜单 & 页面渲染 =====================
menu = st.sidebar.selectbox(
    "功能菜单",
    ["实体查询", "病症找药", "🤖 AI智能问答"]
)

# ========== 实体查询页面（调整按钮：移除【该病症对应药材】） ==========
if menu == "实体查询":
    st.subheader("📌 药材/病症 模板查询 & 模糊检索")
    input_text = st.text_input("请输入药材/病症名称（支持模糊关键词，例：香、腹痛）", value="丁香")
    st.divider()

    # 两行布局，共5个按钮（原第4个按钮移除，分工到独立页面）
    col1, col2, col3 = st.columns(3)
    with col1:
        btn_entity_info = st.button("查询实体完整属性")
    with col2:
        btn_entity_relation = st.button("查询实体关联关系")
    with col3:
        btn_drug_treat = st.button("该药材可治病症")

    col4, col5 = st.columns(2)
    with col4:
        btn_fuzzy = st.button("🔍 全局模糊检索")
    with col5:
        btn_clear = st.button("清空结果")

    st.divider()
    result_props = None
    result_rels = None
    result_fuzzy = None
    input_text = input_text.strip()

    # 按钮事件逻辑
    if btn_entity_info:
        if not input_text:
            st.warning("⚠️ 请输入查询名称！")
        else:
            result_props, result_rels = get_entity_info(input_text)
    elif btn_entity_relation:
        if not input_text:
            st.warning("⚠️ 请输入查询名称！")
        else:
            _, result_rels = get_entity_info(input_text)
    elif btn_drug_treat:
        if not input_text:
            st.warning("⚠️ 请输入查询名称！")
        else:
            result_fuzzy = query_disease_by_herb(input_text)
    elif btn_fuzzy:
        if not input_text:
            st.warning("⚠️ 请输入关键词！")
        else:
            result_fuzzy = fuzzy_search_all(input_text)
    elif btn_clear:
        result_props = None
        result_rels = None
        result_fuzzy = None
        st.info("✅ 已清空所有查询结果")

    # 结果展示
    if result_props and isinstance(result_props, dict):
        st.markdown("### 基本属性")
        st.dataframe(pd.DataFrame(list(result_props.items()), columns=["属性", "值"]), use_container_width=True)
    if result_rels and isinstance(result_rels, list):
        st.markdown("### 关联关系")
        st.dataframe(pd.DataFrame(result_rels), use_container_width=True)
    if result_fuzzy is not None:
        if result_fuzzy.empty:
            st.warning("⚠️ 未查询到相关数据，请更换关键词重试")
        else:
            st.success(f"✅ 共查询到 {len(result_fuzzy)} 条结果")
            st.dataframe(result_fuzzy, use_container_width=True)

# ========== 病症找药页面（专门放置：病症 → 药材 反向查询） ==========
elif menu == "病症找药":
    st.subheader("💊 根据病症查询推荐药材")
    st.info("功能说明：输入病症名称，查询可治疗该病症的所有中药材")
    disease = st.text_input("输入病症名称（如：肾虚阳痿）", "肾虚阳痿")
    if st.button("查询"):
        df = query_herbs_for_disease(disease)
        if df.empty:
            st.warning("未找到相关药材")
        else:
            st.success(f"找到{len(df)}种相关药材")
            st.dataframe(df, use_container_width=True)

# ========== AI智能问答页面 ==========
elif menu == "🤖 AI智能问答":
    st.subheader("🤖 中医药AI问答")
    st.info("结合知识图谱回答药材、病症相关问题，不编造内容")
    user_question = st.text_area("请输入问题", placeholder="例如：丁香的性味归经是什么？可以治疗哪些病症？", height=100)
    if st.button("开始问答", type="primary") and user_question.strip():
        with st.spinner("正在检索图谱并思考..."):
            graph_ctx = search_graph_context(user_question)
            with st.expander("📚 知识图谱检索详情", expanded=False):
                st.write(graph_ctx)
            try:
                answer = call_tongyi_api(DASHSCOPE_API_KEY, graph_ctx, user_question)
                st.markdown("### ✅ AI 回答")
                st.write(answer)
            except Exception as e:
                st.error(f"调用大模型失败：{str(e)}")
