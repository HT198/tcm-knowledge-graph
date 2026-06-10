import streamlit as st
from neo4j import GraphDatabase
import pandas as pd

# --------------------------
# 配置页面
# --------------------------
st.set_page_config(page_title="中医药知识图谱", layout="wide")
st.title("🌿 中医药知识图谱查询系统")

# --------------------------
# 连接 Neo4j Aura（用Streamlit Secrets）
# --------------------------
@st.cache_resource
def init_driver():
    uri = st.secrets["neo4j_uri"]
    user = st.secrets["neo4j_user"]
    pwd = st.secrets["neo4j_password"]
    return GraphDatabase.driver(uri, auth=(user, pwd))

driver = init_driver()

# --------------------------
# 核心查询函数
# --------------------------
def get_entity_info(entity_name):
    with driver.session() as session:
        # 实体基本信息（适配你当前的标签：nodes.csv）
        res = session.run("""
            MATCH (n:`nodes.csv` {id: $name}) RETURN n
        """, name=entity_name)
        record = res.single()
        if not record:
            return None, None
        node = record["n"]
        # 提取属性
        props = {k: v for k, v in node.items() if k not in ["id", "label"]}
        # 提取关系
        relations = session.run("""
            MATCH (n:`nodes.csv` {id: $name})-[r]-(m)
            RETURN type(r) AS 关系类型, m.id AS 关联实体
        """, name=entity_name).data()
    return props, relations

def query_herbs_for_disease(disease_name):
    with driver.session() as session:
        res = session.run("""
            MATCH (m:`nodes.csv`)-[r]->(d:`nodes.csv` {id: $disease})
            WHERE type(r) IN ['治疗', '含有成分']
            RETURN DISTINCT m.id AS 推荐药材
        """, disease=disease_name)
    return pd.DataFrame(res.data())

# --------------------------
# 侧边栏菜单
# --------------------------
menu = st.sidebar.selectbox("功能菜单", ["实体查询", "病症找药"])

# --------------------------
# 页面1：实体查询
# --------------------------
if menu == "实体查询":
    st.subheader("📌 药材/实体查询")
    entity_name = st.text_input("输入实体名称（如：丁香）", "丁香")
    if st.button("查询"):
        props, relations = get_entity_info(entity_name)
        if props is None:
            st.warning("未找到该实体，请检查名称是否正确")
        else:
            # 显示属性
            st.markdown("### 基本属性")
            st.dataframe(pd.DataFrame(list(props.items()), columns=["属性", "值"]), use_container_width=True)
            # 显示关系
            if relations:
                st.markdown("### 关联关系")
                st.dataframe(pd.DataFrame(relations), use_container_width=True)
            else:
                st.info("该实体暂无关联关系")

# --------------------------
# 页面2：病症找药
# --------------------------
elif menu == "病症找药":
    st.subheader("💊 根据病症查询推荐药材")
    disease = st.text_input("输入病症名称（如：脾胃虚寒）", "脾胃虚寒")
    if st.button("查询"):
        df = query_herbs_for_disease(disease)
        if df.empty:
            st.warning("未找到相关药材")
        else:
            st.success(f"找到{len(df)}种相关药材")
            st.dataframe(df, use_container_width=True)
