import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.evaluation.qa import QAEvalChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

# Page config
st.set_page_config(page_title="📊 Sales Insights Dashboard", layout="wide")

# Load and process data
@st.cache_data
def load_data():
    df = pd.read_csv("sales_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    df['Quarter'] = df['Date'].dt.to_period('Q')
    df['Weekday'] = df['Date'].dt.day_name()
    df['Age_Group'] = pd.cut(df['Customer_Age'], bins=[0, 25, 35, 50, 100], labels=['<25', '26-35', '36-50', '50+'])
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔎 Filters")
region_filter = st.sidebar.multiselect("Select Region:", df['Region'].unique(), default=df['Region'].unique())
product_filter = st.sidebar.multiselect("Select Product:", df['Product'].unique(), default=df['Product'].unique())
gender_filter = st.sidebar.multiselect("Select Gender:", df['Customer_Gender'].unique(), default=df['Customer_Gender'].unique())

filtered_df = df[
    df['Region'].isin(region_filter) &
    df['Product'].isin(product_filter) &
    df['Customer_Gender'].isin(gender_filter)
]

# Tabs for page navigation
tabs = st.tabs(["📈 Dashboard", "🧠 QA Evaluation", "📊 Stats Summary", "🤖 BI Assistant"])

# --- Tab 1: Dashboard ---
with tabs[0]:
    st.title("📈 Interactive Business Intelligence Dashboard")

    st.markdown("### 📅 Sales Trends Over Time")
    sales_trend = filtered_df.groupby('Month')['Sales'].sum()
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sales_trend.plot(ax=ax1, marker='o', color='dodgerblue')
    ax1.set_title("Monthly Sales Trend")
    ax1.set_ylabel("Sales")
    ax1.set_xlabel("Month")
    ax1.grid(True)
    st.pyplot(fig1)

    st.markdown("### 📦 Product Performance")
    product_perf = filtered_df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
    fig2, ax2 = plt.subplots()
    product_perf.plot(kind='bar', ax=ax2, color='orange')
    ax2.set_title("Total Sales by Product")
    ax2.set_ylabel("Sales")
    ax2.set_xlabel("Product")
    st.pyplot(fig2)

    st.markdown("### 🌍 Regional Sales Analysis")
    region_perf = filtered_df.groupby('Region')['Sales'].sum()
    fig3, ax3 = plt.subplots()
    region_perf.plot(kind='barh', ax=ax3, color='mediumseagreen')
    ax3.set_title("Sales by Region")
    ax3.set_xlabel("Sales")
    st.pyplot(fig3)

    st.markdown("### 👥 Customer Demographics and Segmentation")
    gender_age_group = filtered_df.groupby(['Age_Group', 'Customer_Gender'])['Sales'].mean().unstack()
    fig4, ax4 = plt.subplots()
    gender_age_group.plot(kind='bar', ax=ax4)
    ax4.set_title("Avg Sales by Age Group & Gender")
    ax4.set_ylabel("Average Sales")
    ax4.set_xlabel("Age Group")
    ax4.legend(title="Gender")
    st.pyplot(fig4)

    st.success("✅ Dashboard generated with dynamic filters and visual analytics.")

# --- Tab 2: QA Evaluation ---
with tabs[1]:
    st.title("🧠 QA Evaluation Results")
    st.markdown("This section displays evaluation results of the LLM-based assistant.")

    class StatRetriever:
        def __init__(self, df):
            self.df = df

        def retrieve(self, query: str) -> str:
            query = query.lower()
            if "monthly sales" in query:
                return self.df.groupby('Month')['Sales'].sum().to_string()
            elif "quarterly sales" in query or "q1" in query or "q2" in query:
                return self.df.groupby('Quarter')['Sales'].sum().to_string()
            elif "region" in query:
                return self.df.groupby('Region')['Sales'].sum().to_string()
            elif "product" in query and "sales" in query:
                return self.df.groupby('Product')['Sales'].sum().to_string()
            elif "customer satisfaction" in query:
                return self.df.groupby('Product')['Customer_Satisfaction'].agg(['mean', 'median', 'std']).to_string()
            elif "segmentation" in query:
                return self.df.groupby(['Age_Group', 'Customer_Gender'])['Sales'].agg(['mean', 'count']).to_string()
            elif "median" in query:
                return f"Median sales: {self.df['Sales'].median()}"
            else:
                return "No matching stats found."

    def evaluate_with_qa_eval_chain(test_df, qa_chain, judge_model, stat_retriever):
        eval_chain = QAEvalChain.from_llm(judge_model)
        examples = []
        predictions = []
        for _, row in test_df.iterrows():
            question = row["question"]
            reference = row["reference"]
            stats = stat_retriever.retrieve(question)
            prompt = f"""
            You are a business intelligence assistant.
            Use the following data to answer the question.

            --- Data ---
            {stats}

            --- Question ---
            {question}
            """
            answer = qa_chain.run(prompt)
            examples.append({"query": question, "reference": reference, "answer": answer})
            predictions.append({"result": answer})
        grades = eval_chain.evaluate(examples=examples, predictions=predictions)
        results = []
        for example, pred_dict, grade in zip(examples, predictions, grades):
            results.append({
                "question": example["query"],
                "reference_answer": example["reference"],
                "prediction": pred_dict["result"],
                "evaluation": grade["results"]
            })
        return pd.DataFrame(results)

    @st.cache_resource
    def setup_rag_chain():
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory="vectorstore", embedding_function=embeddings)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        llm = LlamaCpp(
            model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            temperature=0.2,
            max_tokens=1024,
            n_ctx=4096,
            verbose=False,
            n_threads=8
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return qa_chain, llm, memory

    qa_chain, judge_model, _ = setup_rag_chain()
    stat_retriever = StatRetriever(df)

    uploaded_file = st.file_uploader("📄 Upload a CSV file with `question,reference` columns", type="csv")
    if uploaded_file:
        test_df = pd.read_csv(uploaded_file)
    else:
        st.info("Using sample data (3 test cases)...")
        test_df = pd.DataFrame([
               {"question": "What is the average customer satisfaction for Widget A?", "prediction": "3.014", "reference": "Around 3.014  with low standard deviation.", "evaluation": "CORRECT"},
        {"question": "Which region had the highest sales?", "prediction": "West", "reference": "The region with the highest total sales is West .", "evaluation": "CORRECT"},
        {"question": "What is the median sales amount?", "prediction": "552.5", "reference": "The median sales is approximately 552 units.", "evaluation": "CORRECT"}
        ])

    if st.button("🚀 Run Evaluation"):
        with st.spinner("Evaluating..."):
            results_df = evaluate_with_qa_eval_chain(test_df, qa_chain, judge_model, stat_retriever)
        st.success("✅ Evaluation Complete!")

        st.dataframe(results_df, use_container_width=True)
        st.bar_chart(results_df["evaluation"].value_counts())
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results as CSV", csv, "evaluation_results.csv", "text/csv")

# --- Tab 3: Stats Summary ---
with tabs[2]:
    st.title("📊 Summary Statistics")
    st.markdown("Summary stats of the filtered dataset.")
    st.subheader("🔢 Descriptive Stats")
    st.write(filtered_df.describe())
    st.subheader("📌 Sample Records")
    st.write(filtered_df.head())

# --- Tab 4: BI Assistant ---
with tabs[3]:
    st.title("🤖 InsightForge: AI-Powered BI Assistant")
    st.markdown("Ask about sales, customers, performance. The assistant remembers what you’ve asked earlier.")
    qa_chain, _, memory = setup_rag_chain()
    retriever = StatRetriever(df)
    def generate_chained_response(query, retriever, qa_chain):
        stats = retriever.retrieve(query)
        chat_summary = ""
        if qa_chain.memory.chat_memory.messages:
            for msg in qa_chain.memory.chat_memory.messages:
                if msg.type == "human":
                    chat_summary += f"User: {msg.content}\n"
                else:
                    chat_summary += f"AI: {msg.content}\n"
        insight_prompt = f"""
        You are a business intelligence expert.

        Here is the previous chat history:
        {chat_summary}

        Use the following data to answer the current question.

        --- Data ---
        {stats}

        --- Question ---
        {query}

        --- Insight ---"""
        insight = qa_chain.run(insight_prompt)
        action_prompt = f"""
        Based on this insight: "{insight}",
        give a specific and realistic business recommendation.
        """
        recommendation = qa_chain.run(action_prompt)
        return insight, recommendation

    query = st.text_input("🧠 Ask your business question:")
    if query:
        with st.spinner("Thinking..."):
            insight, recommendation = generate_chained_response(query, retriever, qa_chain)
        st.markdown("### 🔍 Insight")
        st.success(insight)
        st.markdown("### 🎯 Recommendation")
        st.info(recommendation)
    if st.checkbox("🕓 Show chat history"):
        st.markdown("### 💬 Previous Interactions")
        for msg in memory.chat_memory.messages:
            role = "🧑‍💼 You" if msg.type == "human" else "🤖 InsightForge"
            st.markdown(f"**{role}:** {msg.content}")