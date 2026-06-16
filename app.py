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

# ---------------------- 2. 修复后的实体查询函数 ----------------------
def get_entity_info(entity_name):
    with driver.session() as session:
        res = session.run("""
            MATCH (n:Entity {id: $name}) RETURN n
        """, name=entity_name)
        records = list(res)
        if not records:
            return None, None
        
        node = records[0]["n"]
        # 过滤掉空值，只保留有数据的属性，不会再显示 None
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

# ---------------------- 3. 查询函数 ----------------------
# 药材 -> 可治疗病症
def query_disease_by_herb(herb_name):
    with driver.session() as session:
        res = session.run("""
            MATCH (m:Entity {id: $name})-[r]->(d:Entity)
            WHERE type(r) = '治疗'
            RETURN DISTINCT d.id AS 对应病症
        """, name=herb_name)
        records = list(res)
        data = [rec.data() for rec in records]
        return pd.DataFrame(data)

# 病症 -> 对应药材
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

# ---------------------- 4.全局模糊查询 ----------------------
def fuzzy_search_all(keyword):
    try:
        with driver.session() as session:
            cypher = """
            MATCH (n:Entity)
            WHERE n.id CONTAINS $kw
            RETURN DISTINCT n.id AS 实体名称 LIMIT 50
            """
            res = session.run(cypher, kw=keyword)
            records = list(res)
            data = [rec.data() for rec in records]
            return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()

# ---------------------- 5.病症专属模糊查询 ----------------------
def fuzzy_disease_with_herb(keyword):
    with driver.session() as session:
        cypher = """
        MATCH (m:Entity)-[r]->(d:Entity)
        WHERE type(r) = '治疗' AND d.id CONTAINS $kw
        RETURN DISTINCT d.id AS 病症名称, m.id AS 对应药材 LIMIT 100
        """
        res = session.run(cypher, kw=keyword)
        records = list(res)
        data = [rec.data() for rec in records]
        return pd.DataFrame(data)

# ---------------------- 6.大模型调用函数 ----------------------
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

# ========= 纯大模型问答（不读取Neo4j图谱，仅通用中医药回答） =========
def call_raw_llm(api_key, user_question):
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    prompt = f"""你是资深中医药科普专家，无数据库限制，可依托通用中医药知识库自由作答。
允许完整科普药材、病症、方剂、养生知识，无需限制图谱数据，回答通俗易懂，条理清晰。
若涉及用药建议，末尾统一标注：中药使用请遵循专业中医师指导，请勿自行抓药服用。

用户提问：{user_question}
"""
    payload = {
        "model": "qwen-turbo",
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "parameters": {"temperature": 0.7, "max_tokens": 2000}
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    return response.json()["output"]["text"]

# ---------------------- 7.图谱上下文检索 ----------------------
def search_graph_context(question):
    context = []
    q_clean = question.replace("？", "").replace("什么是", "").replace("是什么", "").strip()
    with driver.session() as session:
        # 提取所有2字及以上有效名词片段，过滤虚词
        import re
        # 提取中文实词
        words = re.findall(r'[\u4e00-\u9fa5]{2,}', q_clean)
        # 兜底单字药材
        single_chars = re.findall(r'[\u4e00-\u9fa5]', q_clean)
        keywords = list(set(words + single_chars))

        entity_ids = set()
        # 循环匹配每个关键词
        for kw in keywords:
            res = session.run("""
                MATCH (n:Entity)
                WHERE n.id = $kw OR n.id CONTAINS $kw
                RETURN DISTINCT n.id
            """, kw=kw)
            for rec in res:
                entity_ids.add(rec["n.id"])
        entity_ids = list(entity_ids)

        # 拼接实体属性与关系
        if entity_ids:
            context.append(f"✅ 匹配到的实体：{', '.join(entity_ids)}")
            for entity_id in entity_ids[:3]:
                res_node = session.run("MATCH (n:Entity {id: $id}) RETURN n", id=entity_id)
                node = list(res_node)[0]["n"]
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
                    context.append(f"🔗 {s} —[{r}]→ {t}")

        # 全局关系兜底检索
        for kw in keywords:
            res_rel = session.run("""
                MATCH (s:Entity)-[r]->(t:Entity)
                WHERE s.id CONTAINS $kw OR t.id CONTAINS $kw
                RETURN s.id, type(r), t.id
            """, kw=kw)
            for rec in res_rel:
                s, r, t = rec.values()
                context.append(f"🔍 关联关系：{s} —[{r}]→ {t}")

    return "\n".join(context) if context else "知识图谱中未查询到相关信息"
# ---------------------- 8. 页面菜单 ----------------------
menu = st.sidebar.selectbox(
    "功能菜单",
    ["实体查询", "病症找药", "🤖 AI智能问答"]
)

# 实体查询页面
if menu == "实体查询":
    st.subheader("📌 药材/病症 查询")
    # 统一输入框
    input_text = st.text_input("请输入药材/病症名称")
    st.divider()

    # 两行布局，移除原第六个按钮，保留5个按钮
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
    # 结果容器
    result_props = None
    result_rels = None
    result_fuzzy = None

    # 按钮事件逻辑
    if btn_entity_info and input_text.strip():
        result_props, _ = get_entity_info(input_text)
    elif btn_entity_relation and input_text.strip():
        _, result_rels = get_entity_info(input_text)
    elif btn_drug_treat and input_text.strip():
        # 调用修复后的药材查病症函数
        result_fuzzy = query_disease_by_herb(input_text)
    elif btn_fuzzy and input_text.strip():
        result_fuzzy = fuzzy_search_all(input_text)
    elif btn_clear:
        result_props = None
        result_rels = None
        result_fuzzy = None
        st.info("✅ 已清空所有查询结果")

    # ========== 结果展示区 ==========
    # 1. 实体属性展示
    if result_props is not None:
        st.markdown("### 基本属性")
        st.dataframe(pd.DataFrame(list(result_props.items()), columns=["属性", "值"]), use_container_width=True)
    # 2. 关联关系展示
    if result_rels is not None:
        st.markdown("### 关联关系")
        if result_rels:
            st.dataframe(pd.DataFrame(result_rels), use_container_width=True)
        else:
            st.info("该实体暂无关联关系")
    # 3. 模糊查询/病症结果展示
    if result_fuzzy is not None:
        if result_fuzzy.empty:
            st.warning("⚠️ 未查询到相关数据，请更换关键词重试")
        else:
            st.success(f"✅ 共查询到 {len(result_fuzzy)} 条结果")
            st.dataframe(result_fuzzy, use_container_width=True)

    # 空输入校验
    if not input_text.strip() and (btn_entity_info or btn_entity_relation or btn_fuzzy):
        st.warning("⚠️ 请先输入查询内容！")

# 病症找页面
elif menu == "病症找药":
    st.subheader("💊 根据病症查询推荐药材")
    disease = st.text_input("输入病症名称")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        btn_search = st.button("精准查询")
    with col2:
        btn_fuzzy_dis = st.button("🔍 模糊检索")

    df_result = None
    if btn_search and disease.strip():
        df_result = query_herbs_for_disease(disease)
    elif btn_fuzzy_dis and disease.strip():
        df_result = fuzzy_disease_with_herb(disease)

    if df_result is not None:
        if df_result.empty:
            st.warning("未找到相关药材")
        else:
            st.success(f"找到{len(df_result)}种相关药材")
            st.dataframe(df_result, use_container_width=True)

# AI智能问答页面
elif menu == "🤖 AI智能问答":
    st.subheader("🤖 AI智能问答")
    user_question = st.text_area("请输入问题", height=100)
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
elif menu == "✨ 纯大模型问答":
    st.subheader("✨ 纯大模型通用问答")
    st.info("不受图谱数据限制，可自由科普各类中医药知识，适合宽泛科普提问")
    raw_question = st.text_area("输入你的中医药问题", height=100)
    if st.button("直接生成回答", type="primary") and raw_question.strip():
        with st.spinner("大模型正在生成科普回答..."):
            try:
                api_key = st.secrets["DASHSCOPE_API_KEY"]
                raw_ans = call_raw_llm(api_key, raw_question)
                st.markdown("### 📖 纯模型科普回答")
                st.write(raw_ans)
            except Exception as e:
                st.error(f"大模型调用失败：{str(e)}")
