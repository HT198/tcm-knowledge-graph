import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 页面基础配置
st.set_page_config(page_title="中医药知识图谱+AI问答", layout="wide")
st.title("🌿 中医药知识图谱智能系统（AI版）")

# ---------------------- 1. 连接Neo4j数据库 ----------------------
@st.cache_resource
def init_driver():
    uri = "neo4j+s://cb5cc04e.databases.neo4j.io"
    user = "cb5cc04e"
    pwd = "uzjqMbskUGdIObBV0uRRs6AoTO8gpmetKkHyhd3vuhs"
    return GraphDatabase.driver(uri, auth=(user, pwd))

driver = init_driver()

# ---------------------- 2. 初始化大模型（从Secrets读取密钥） ----------------------
@st.cache_resource
def init_llm_chain():
    # 从Streamlit密钥读取API Key，安全不泄露
    api_key = st.secrets["DASHSCOPE_API_KEY"]
    llm = ChatOpenAI(
        model="qwen-turbo",
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.3,
        max_tokens=2000
    )
    prompt = PromptTemplate(
        input_variables=["graph_context", "user_question"],
        template="""
你是专业中医药顾问，请严格依据【知识图谱检索结果】回答问题。
如果没有相关信息，请如实说明，严禁编造内容。

【知识图谱检索结果】：
{graph_context}

【用户问题】：
{user_question}

请用通俗、条理清晰的语言作答：
"""
    )
    return LLMChain(llm=llm, prompt=prompt)

llm_chain = init_llm_chain()

# ---------------------- 3. 原有图谱查询函数 ----------------------
def get_entity_info(entity_name):
    with driver.session() as session:
        res = session.run("MATCH (n:Entity {id: $name}) RETURN n", name=entity_name)
        records = list(res)
        if not records:
            return None, None
        node = records[0]["n"]
        props = dict(node)

        res_rel = session.run("""
            MATCH (n:Entity {id: $name})-[r]-(m)
            RETURN type(r) AS 关系类型, m.id AS 关联实体
        """, name=entity_name)
        rel_list = list(res)
        relations = [rec.data() for rec in rel_list]
    return props, relations

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

# ---------------------- 4. 图谱上下文检索（给AI提供素材） ----------------------
def search_graph_context(question):
    context = []
    with driver.session() as session:
        # 模糊匹配相关实体
        entity_res = session.run("""
            MATCH (n:Entity) WHERE n.id CONTAINS $kw RETURN n.id LIMIT 10
        """, kw=question)
        entity_list = [rec["n.id"] for rec in list(entity_res)]
        if entity_list:
            context.append(f"匹配实体：{', '.join(entity_list)}")
        # 检索实体关联关系
        for entity in entity_list[:3]:
            rel_res = session.run("""
                MATCH (n:Entity{id:$e})-[r]-(m)
                RETURN n.id, type(r), m.id LIMIT 15
            """, e=entity)
            for rec in list(rel_res):
                s, r, t = rec.values()
                context.append(f"{s} —{r}→ {t}")
    return "\n".join(context) if context else "知识图谱中未查询到相关信息"

# ---------------------- 5. 页面功能 ----------------------
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
            st.dataframe(pd.DataFrame(list(props.items()), columns=["属性", "值"]), use_container_width=True)
            if relations:
                st.markdown("### 关联关系")
                st.dataframe(pd.DataFrame(relations), use_container_width=True)
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
            st.dataframe(df, use_container_width=True)

# AI智能问答
elif menu == "🤖 AI智能问答":
    st.subheader("🤖 中医药AI问答")
    st.info("结合知识图谱回答药材、病症相关问题")
    user_question = st.text_area("请输入问题", placeholder="例如：丁香可以治疗哪些病症？", height=100)
    if st.button("开始问答", type="primary") and user_question.strip():
        with st.spinner("正在检索图谱并思考..."):
            graph_ctx = search_graph_context(user_question)
            with st.expander("📚 知识图谱检索详情", expanded=False):
                st.write(graph_ctx)
            # 调用大模型
            answer = llm_chain.invoke({"graph_context": graph_ctx, "user_question": user_question})["text"]
            st.markdown("### ✅ AI 回答")
            st.write(answer)
