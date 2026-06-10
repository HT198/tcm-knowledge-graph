import streamlit as st
from neo4j import GraphDatabase
import pandas as pd

# 页面配置
st.set_page_config(page_title="中医药知识图谱", layout="wide")
st.title("🌿 中医药知识图谱查询系统")

# 数据库连接
@st.cache_resource
def init_driver():
    uri = "neo4j+s://cb5cc04e.databases.neo4j.io"
    user = "cb5cc04e"
    pwd = "uzjqMbskUGdIObBV0uRRs6AoTO8gpmetKkHyhd3vuhs"
    return GraphDatabase.driver(uri, auth=(user, pwd))

driver = init_driver()

# 实体查询（修复结果读取异常）
def get_entity_info(entity_name):
    with driver.session() as session:
        res = session.run("""
            MATCH (n:Entity {id: $name}) RETURN n
        """, name=entity_name)
        # 先转列表，防止迭代器重复读取
        records = list(res)
        if not records:
            return None, None
        
        node = records[0]["n"]
        props = dict(node)
        
        # 查询关联关系
        res_rel = session.run("""
            MATCH (n:Entity {id: $name})-[r]-(m)
            RETURN type(r) AS 关系类型, m.id AS 关联实体
        """, name=entity_name)
        rel_list = list(res_rel)
        relations = [rec.data() for rec in rel_list]
    return props, relations

# 病症查药材（兼容所有治疗类关系 + 修复读取异常）
def query_herbs_for_disease(disease_name):
    with driver.session() as session:
        res = session.run("""
            MATCH (m:Entity)-[r]->(d:Entity {id: $disease})
            WHERE type(r) = '治疗'
            RETURN DISTINCT m.id AS 推荐药材
        """, disease=disease_name)
        # 核心修复：先转为列表再解析
        records = list(res)
        data = [rec.data() for rec in records]
        return pd.DataFrame(data)

# 侧边菜单
menu = st.sidebar.selectbox("功能菜单", ["实体查询", "病症找药"])

# 实体查询页面
if menu == "实体查询":
    st.subheader("📌 药材/实体查询")
    entity_name = st.text_input("输入实体名称（如：丁香）", "丁香")
    if st.button("查询"):
        props, relations = get_entity_info(entity_name)
        if props is None:
            st.warning("未找到该实体，请检查名称是否正确")
        else:
            st.markdown("### 基本属性")
            df_props = pd.DataFrame(list(props.items()), columns=["属性", "值"])
            st.dataframe(df_props, use_container_width=True)
            
            if relations:
                st.markdown("### 关联关系")
                st.dataframe(pd.DataFrame(relations), use_container_width=True)
            else:
                st.info("该实体暂无关联关系")

# 病症找药页面
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
