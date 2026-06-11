import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
import requests
import json

# 页面配置
st.set_page_config(page_title="中医药知识图谱", layout="wide")
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

# ---------------------- 5. 新增：图谱上下文检索（给AI用，不影响原有查询） ----------------------
def search_graph_context(question):
    context = []
    with driver.session() as session:
        # ---------------------- 1. 先精准匹配实体（比如“橘红”） ----------------------
        # 优先做完全匹配，避免模糊匹配遗漏
        entity_res = session.run("""
            MATCH (n:Entity) WHERE n.id = $kw OR n.id CONTAINS $kw RETURN n
        """, kw=question)
        entities = list(entity_res)
        if entities:
            context.append(f"✅ 匹配到的实体：{', '.join([rec['n']['id'] for rec in entities])}")

            # 对每个实体，拉取【所有属性】和【所有关联关系】
            for rec in entities[:3]:
                node = rec["n"]
                node_id = node["id"]

                # 提取实体属性（性味、归经、检测相关等）
                props_str = ", ".join([f"{k}:{v}" for k, v in dict(node).items() if v])
                if props_str:
                    context.append(f"📋 {node_id} 属性：{props_str}")

                # 提取实体的所有关联关系（不管方向、不管关系名）
                rel_res = session.run("""
                    MATCH (a:Entity)-[r]-(b:Entity)
                    WHERE a.id = $id OR b.id = $id
                    RETURN a.id, type(r), b.id
                """, id=node_id)
                for rel_rec in rel_res:
                    s, r, t = rel_rec.values()
                    context.append(f"🔗 图谱关系：{s} —[{r}]→ {t}")

        # ---------------------- 2. 反向匹配：所有和问题关键词相关的关系 ----------------------
        # 处理“用什么检测橘红”这种问题，匹配所有指向/包含“橘红”的关系
        rel_res = session.run("""
            MATCH (s:Entity)-[r]->(t:Entity)
            WHERE s.id CONTAINS $kw OR t.id CONTAINS $kw
            RETURN s.id, type(r), t.id
        """, kw=question)
        for rec in rel_res:
            s, r, t = rec.values()
            context.append(f"🔍 关联关系：{s} —[{r}]→ {t}")

    # 如果还是没数据，直接返回无结果
    if not context:
        return "知识图谱中未查询到相关信息"
    return "\n".join(context)

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
