import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
import requests
import json

# 页面配置
st.set_page_config(page_title="中医药知识图谱+AI问答", layout="wide")
st.title("🌿 中医药知识图谱智能系统")

# ---------------------- 1. 数据库连接 ----------------------
@st.cache_resource
def init_driver():
    uri = "neo4j+s://cb5cc04e.databases.neo4j.io"
    user = "cb5cc04e"
    pwd = "uzjqMbskUGdIObBV0uRRs6AoTO8gpmetKkHyhd3vuhs"
    return GraphDatabase.driver(uri, auth=(user, pwd))

driver = init_driver()

# ---------------------- 2. 修复后的实体查询函数（正确显示属性） ----------------------
def get_entity_info(entity_name):
    with driver.session() as session:
        res = session.run("""
            MATCH (n:Entity {id: $name}) RETURN n
        """, name=entity_name)
        records = list(res)
        if not records:
            return None, None
        
        node = records[0]["n"]
        # 核心修复：过滤掉空值，只保留有数据的属性，不会再显示 None
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

# ---------------------- 3. 原有病症查询函数（不变） ----------------------
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

# ---------------------- 4. 大模型调用函数（新增，不影响原有功能） ----------------------
def call_tongyi_api(api_key, graph_context, user_question):
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    prompt = f"""你是专业的中医药顾问，必须严格遵循以下规则回答问题：
1.  优先依据【知识图谱检索结果】中的信息作答，图谱中提到的药材必须优先使用。
2.  如果图谱中有明确的【治疗】关系（如“八角茴香 → 治疗 → 肾虚腰痛”），必须以该药材为核心回答。
3.  禁止编造图谱中没有的药材、方剂（如金匮肾气丸、六味地黄丸等）。
4.  图谱中提到的药材，可以结合它的属性（性味、归经、功能主治）补充说明。
5.  如果图谱无数据，可说明情况并建议咨询专业中医师。

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

# ---------------------- 5. 图谱上下文检索（给AI用） ----------------------
def search_graph_context(question):
    context = []
    with driver.session() as session:
        # ---------------------- 1. 从问题中提取关键词（简单分词） ----------------------
        keywords = []
        # 把问题按空格/标点拆分，提取有意义的词
        for token in question.replace("？", "").replace("用什么", "").replace("什么药", "").split():
            if len(token) > 1:  # 过滤掉单字虚词
                keywords.append(token)
        # 加上原问题文本，兜底匹配
        keywords.append(question)

        # ---------------------- 2. 匹配所有相关实体（含模糊匹配） ----------------------
        entity_ids = set()
        for kw in keywords:
            # 优先精准匹配，再模糊匹配
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

            # ---------------------- 3. 对每个实体，拉取【属性】和【所有进出关系】 ----------------------
            for entity_id in entity_ids[:3]:  # 取前3个，避免上下文过长
                # 3.1 提取实体属性
                res = session.run("""
                    MATCH (n:Entity {id: $id}) RETURN n
                """, id=entity_id)
                node = list(res)[0]["n"]
                props_str = ", ".join([f"{k}:{v}" for k, v in dict(node).items() if v])
                if props_str:
                    context.append(f"📋 {entity_id} 属性：{props_str}")

                # 3.2 提取所有关系（不管方向、不管关系名）
                res_rel = session.run("""
                    MATCH (a:Entity)-[r]-(b:Entity)
                    WHERE a.id = $id OR b.id = $id
                    RETURN a.id, type(r), b.id
                """, id=entity_id)
                for rec in res_rel:
                    s, r, t = rec.values()
                    context.append(f"🔗 图谱关系：{s} —[{r}]→ {t}")

        # ---------------------- 4. 反向兜底：匹配所有包含关键词的关系 ----------------------
        for kw in keywords:
            res = session.run("""
                MATCH (s:Entity)-[r]->(t:Entity)
                WHERE s.id CONTAINS $kw OR t.id CONTAINS $kw
                RETURN s.id, type(r), t.id
            """, kw=kw)
            for rec in res:
                s, r, t = rec.values()
                context.append(f"🔍 关联关系：{s} —[{r}]→ {t}")

    return "\n".join(context) if context else "知识图谱中未查询到相关信息"

# ---------------------- 6. 页面菜单 ----------------------
menu = st.sidebar.selectbox(
    "功能菜单",
    ["实体查询", "病症找药", "🤖 AI智能问答"]
)

# 实体查询页面（修复后）
if menu == "实体查询":
    st.subheader("📌 药材/实体查询")
    entity_name = st.text_input("输入实体名称（如：丁香）", "丁香")
    if st.button("查询"):
        props, relations = get_entity_info(entity_name)
        if props is None:
            st.warning("未找到该实体，请检查名称是否正确")
        else:
            st.markdown("### 基本属性")
            # 只显示有数据的属性，不会再出现 None
            st.dataframe(pd.DataFrame(list(props.items()), columns=["属性", "值"]), use_container_width=True)
            if relations:
                st.markdown("### 关联关系")
                st.dataframe(pd.DataFrame(relations), use_container_width=True)
            else:
                st.info("该实体暂无关联关系")

# 病症找药页面（不变）
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

# AI智能问答页面（新增）
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
