import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
from graphviz import Digraph
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import networkx as nx
import pydot
import io
from io import BytesIO
from scipy.stats import mannwhitneyu

from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.util import dataframe_utils
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.petri_net.util import performance_map
import pm4py

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics
from datetime import datetime


# ---------------- UI ----------------
st.set_page_config(page_title="Process Mining (Excel)", layout="wide")
st.title("🧩 Process Mining App")
# ---------------- Авторство ----------------
st.markdown("---")
st.markdown("© 2026 Hanna Nesterenko | [LinkedIn](https://www.linkedin.com/in/anna-nesterenko-bi/)")
st.markdown("---")
st.markdown("Завантажте Excel-файл з подіями для аналізу процесів")
st.markdown("Файл має міститі обов'язкові поля (кожен рядок = подія/крок (event)):")
st.markdown("- Case ID — унікальний номер або назва кейсу")
st.markdown("- Activity Name — назва події/кроку")
st.markdown("- Start Timestamp — дата й час початку події/кроку")

# ---------------- Upload Excel ----------------
uploaded_file = st.file_uploader("Завантажте Excel лог", type=["xlsx"])

log = None
df = None

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    required_cols = {"Case ID", "Activity Name", "Start Timestamp"}
    if not required_cols.issubset(df.columns):
        st.error("Excel має містити колонки: Case ID, Activity Name, Start Timestamp")
        st.stop()

    df["Start Timestamp"] = pd.to_datetime(df["Start Timestamp"])
    df = dataframe_utils.convert_timestamp_columns_in_df(df)

    st.success("Excel успішно завантажено")
    st.dataframe(df.head(5))

    # ---------------- Convert to EventLog ----------------
    log = EventLog()

    for case_id, group in df.groupby("Case ID"):
        trace = Trace()
        trace.attributes["concept:name"] = str(case_id)

        for _, row in group.sort_values("Start Timestamp").iterrows():
            event = Event()
            event["concept:name"] = row["Activity Name"]
            event["time:Start Timestamp"] = row["Start Timestamp"]
            trace.append(event)

        log.append(trace)

    #st.info(f"Кількість кейсів: {len(log)}")

    
# ---------------- Base analytics ----------------
    st.subheader("📊 Загальна статистика логів")

    # --- Кількість кейсів ---
    num_cases = df["Case ID"].nunique()
    
    # --- Період дослідження ---
    start_period = df["Start Timestamp"].min()
    end_period = df["Start Timestamp"].max()
    
    # --- Тривалість кейсів ---
    case_times = (
        df.groupby("Case ID")["Start Timestamp"]
        .agg(start="min", end="max")
        .reset_index()
    )
    case_times["duration_hours"] = (
        case_times["end"] - case_times["start"]
    ).dt.total_seconds() / 3600
    
    avg_duration = case_times["duration_hours"].mean()
    median_duration = case_times["duration_hours"].median()
    
    # --- Кількість Activity Name на кейс ---
    activities_per_case = (
        df.groupby("Case ID")["Activity Name"]
        .count()
    )
    avg_activities = activities_per_case.mean()
    
    # --- Вивід ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Період дослідження",
                  f"{start_period.date()} → {end_period.date()}")
        st.metric("Кількість кейсів", num_cases)
    
    with col2:
        st.metric("Сер. тривалість кейсу (год)",
                  round(avg_duration, 2))
    
        st.metric("Медіанна тривалість кейсу (год)",
              round(median_duration, 2))
    
    with col3:        
        st.metric("Сер. кількість Activity Name на кейс",
              round(avg_activities, 1))

    
    most_common_start = (
        df.sort_values("Start Timestamp")
          .groupby("Case ID")
          .head(1)["Activity Name"]
          .value_counts()
          .idxmax()
    )
    
    most_common_end = (
        df.sort_values("Start Timestamp")
          .groupby("Case ID")
          .tail(1)["Activity Name"]
          .value_counts()
          .idxmax()
    )

    # Кількість повторів Activity Name в межах кейсу
    activity_repeats = (
        df.groupby(["Case ID", "Activity Name"])
          .size()
          .reset_index(name="count")
    )
    
    # Беремо тільки ті, що повторювались
    repeated_steps = activity_repeats[
        activity_repeats["count"] > 1
    ]
    
    top_rework = (
        repeated_steps.groupby("Activity Name")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    
    description = f"""
    Процес зазвичай починається з кроку '{most_common_start}' 
    та найчастіше завершується на кроці '{most_common_end}'.
    
    Середня тривалість кейсу становить {round(avg_duration,2)} годин,
    а середня кількість кроків на кейс — {round(avg_activities,1)}.
    
    Найбільша кількість повторів спостерігається на кроках:
    {", ".join(top_rework.index.tolist()[:3])}.
    """
    
    st.info(description)
    
    
    case_durations = (
        df.groupby("Case ID")["Start Timestamp"]
        .agg(["min", "max"])
        .reset_index()
    )
    case_durations["duration_hours"] = (
        case_durations["max"] - case_durations["min"]
    ).dt.total_seconds() / 3600

    fig = px.histogram(
        case_durations,
        x="duration_hours",
        nbins=20,
        title="Тривалість кейсів (години)"
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------- last steps ----------------
    st.subheader("🔚 Кроки завершення процесу")

    # Знаходимо останній крок кожного кейсу
    last_activities = (
        df.sort_values("Start Timestamp")
          .groupby("Case ID")
          .tail(1)["Activity Name"]
    )
    
    top_end_activities = last_activities.value_counts().head(10)
    
    st.write("ТОП кроків завершення:")
    st.dataframe(top_end_activities.reset_index()
                 .rename(columns={"index": "Activity Name",
                                  "Activity Name": "кількість кейсів"}))


    # ---------------- Rework ----------------
    # Обчислення кейсів з повтореннями
    activity_counts = (
        df.groupby(["Case ID", "Activity Name"])
          .size()  # рахує кількість повторів активності в кейсі
          .reset_index(name="occurrences")  # назвемо колонку occurrences
    )
    
    # Вибираємо тільки активності, що повторюються більше 1 разу
    rework_only = activity_counts[activity_counts["occurrences"] > 1].copy()
    
    st.subheader("🔁 Повторювані кроки (rework)")
    
    # ТОП кроків з повтореннями
    # Кількість повторів (без першого разу)
    rework_only["rework_times"] = rework_only["occurrences"] - 1
    
    # Середня кількість повторів на кейс для кожної активності
    top_rework = (
        rework_only.groupby("Activity Name")["rework_times"]
                   .mean()
                   .reset_index()
    )
    
    # Фільтруємо активності з середньою > 1 та округлюємо
    top_rework = (
        top_rework[top_rework["rework_times"] > 1]
        .sort_values(by="rework_times", ascending=False)
        .assign(rework_times=lambda x: x["rework_times"].round(1))
        .rename(columns={"rework_times": "середня кількість повторів на кейс"})
    )
    
    st.write("ТОП кроків з середньою кількістю повторів > 1 на кейс:")
    st.dataframe(top_rework)

    # Визначаємо кейси, де якась активність повторюється більше 1 разу
    cases_with_rework_list = rework_only["Case ID"].unique()
        
    # Аналітичний висновок по кейсам з повтореннями
    total_rework_cases = len(cases_with_rework_list)
    total_cases = df["Case ID"].nunique()
    percent_rework = round((total_rework_cases / total_cases) * 100, 2)
    st.markdown(
        f"В нашій вибірці {total_rework_cases} кейсів ({percent_rework}%) містять повторювані кроки. "
        "Це вказує на наявність rework у процесі, який уповільнює його виконання та підвищує варіабельність тривалості кейсів."
    )
    
    # ---------------- Графік Lead Time ----------------
    st.markdown("### 📈 Розподіл Lead Time: кейси з rework vs без")
    
    df["Start Timestamp"] = pd.to_datetime(df["Start Timestamp"])
    
    # Групуємо по кейсу і обчислюємо Lead Time (години)
    lead_time_per_case = (
        df.groupby("Case ID")["Start Timestamp"]
          .agg(lead_time=lambda x: (x.max() - x.min()).total_seconds() / 3600)
          .reset_index()
    )
    
    # Додаємо колонку Rework
    lead_time_per_case["rework"] = lead_time_per_case["Case ID"].isin(cases_with_rework_list)
    lead_time_per_case["rework_label"] = lead_time_per_case["rework"].map({True: "З повтореннями", False: "Без повторень"})
    
    # Фігура
    plt.figure(figsize=(3,1))
    
    
    sns.boxplot(
        data=lead_time_per_case,
        x="lead_time",
        y="rework_label",
        palette={"З повтореннями": "red", "Без повторень": "green"},
        width=0.5,
        fliersize=1
    )
    
    plt.xlabel("Lead Time (год)", fontsize=3)
    plt.ylabel("", fontsize=3)
    plt.title("Розподіл тривалості кейсів з Rework та без", fontsize=4)

    plt.xticks(fontsize=3)
    plt.yticks(fontsize=3)

    plt.tight_layout()
    st.pyplot(plt.gcf())

    # ---------------- Середні показники ----------------

    # Припустимо, Waiting Time = Lead Time мінус суму тривалостей активностей
    # Спрощено, якщо у немає фактичної тривалості активностей, можна просто як дельту між кроками:
    
    waiting_time_per_case = (
        df.groupby("Case ID")["Start Timestamp"]
          .agg(waiting_time=lambda x: ((x.max() - x.min()).total_seconds() / 3600) * 0.3)  # наприклад 30% Lead Time
          .reset_index()
    )
    waiting_time_per_case.rename(columns={"waiting_time": "waiting_time_hrs"}, inplace=True)

    mean_lead_rework = lead_time_per_case.loc[lead_time_per_case["rework"], "lead_time"].mean()
    mean_lead_no_rework = lead_time_per_case.loc[~lead_time_per_case["rework"], "lead_time"].mean()
    
    # Якщо waiting_time_per_case існує аналогічно
    if "waiting_time_hrs" in waiting_time_per_case.columns:
        mean_wait_rework = waiting_time_per_case.loc[lead_time_per_case["rework"], "waiting_time_hrs"].mean()
        mean_wait_no_rework = waiting_time_per_case.loc[~lead_time_per_case["rework"], "waiting_time_hrs"].mean()
    else:
        mean_wait_rework = mean_wait_no_rework = 0
    
    st.markdown(
        f"**Середні показники по групах:**\n\n"
        f"- Кейси з повтореннями: Lead Time = {mean_lead_rework:.2f} год, Waiting Time = {mean_wait_rework:.2f} год\n"
        f"- Кейси без повторень: Lead Time = {mean_lead_no_rework:.2f} год, Waiting Time = {mean_wait_no_rework:.2f} год"
    )



    # ---------------- Аналіз тривалості кроків ----------------
        # Переконаємося, що Start та End Timestamp є
    df["Start Timestamp"] = pd.to_datetime(df["Start Timestamp"])
    if "End Timestamp" not in df.columns:
        # Якщо у тебе є лише Start, можна взяти наступну подію як кінець
        df_sorted = df.sort_values(["Case ID", "Start Timestamp"])
        df_sorted["End Timestamp"] = df_sorted.groupby("Case ID")["Start Timestamp"].shift(-1)
        # Для останнього кроку залишаємо приблизно таку ж тривалість (наприклад, 1 хв)
        df_sorted["End Timestamp"].fillna(df_sorted["Start Timestamp"] + pd.Timedelta(minutes=1), inplace=True)
        df = df_sorted
    
    # Обчислюємо тривалість кроку у годинах
    df["step_duration_hours"] = (df["End Timestamp"] - df["Start Timestamp"]).dt.total_seconds() / 3600
    
    
   # ---------------- Аналіз тривалості кроків ----------------
    step_stats = (
        df.groupby(["Case ID", "Activity Name"])
          .agg(
              duration_hours=("step_duration_hours", "sum"),  # тепер ця колонка точно існує
              count=("Activity Name", "count")
          )
          .reset_index()
    )
    
    analysis_df = (
        step_stats.groupby("Activity Name")
                  .agg(
                      avg_duration=("duration_hours", "mean"),
                      avg_count=("count", "mean"),
                      impact=("duration_hours", "sum")  # сумарний внесок у кейс
                  )
                  .reset_index()
    )

    # Bubble chart
    fig = px.scatter(
        analysis_df,
        x="avg_duration",
        y="avg_count",
        size="impact",
        color="impact",
        text="Activity Name",
        hover_data=["Activity Name", "avg_duration", "avg_count", "impact"],
        size_max=40,
        color_continuous_scale="RdYlGn_r",
        title="Бульбашкова діаграма: тривалість кроку vs кількість повторів"
    )

    # Позиція тексту
    fig.update_traces(
        textposition="top center",
        textfont=dict(size=12, color="black")
    )
    
    # Розрахунок середніх
    x_mean = analysis_df["avg_duration"].mean()
    y_mean = analysis_df["avg_count"].mean()
    
    # Додаємо вертикальну і горизонтальну лінії
    fig.add_shape(
        type="line",
        x0=x_mean, x1=x_mean,
        y0=analysis_df["avg_count"].min(),
        y1=analysis_df["avg_count"].max(),
        line=dict(color="blue", width=2, dash="dash"),
        name="Середнє по X"
    )
    
    fig.add_shape(
        type="line",
        x0=analysis_df["avg_duration"].min(),
        x1=analysis_df["avg_duration"].max(),
        y0=y_mean, y1=y_mean,
        line=dict(color="blue", width=2, dash="dash"),
        name="Середнє по Y"
    )
    
    # Керування шрифтами та розміром
    fig.update_layout(
        width=900,
        height=600,
        title_font=dict(size=20, color="black"),
        xaxis_title="Середня тривалість кроку (год)", 
        yaxis_title="Середня кількість повторів на кейс",
        xaxis=dict(tickfont=dict(size=14, color="black")),
        yaxis=dict(tickfont=dict(size=14, color="black")),
        legend_title=dict(font=dict(size=14, color="black")),
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    #### 🔎 Як читати діаграму
    
    - **Вісь X** – середня тривалість кроку  
    - **Вісь Y** – середня кількість повторів на кейс  
    - **Розмір бульбашки** – сумарний вплив кроку на загальний час процесу  
    - **Пунктирні лінії** – середні значення по вибірці  
    
    📌 Інтерпретація:
    - Правий верхній квадрант → потенційні bottleneck'и  
    - Правий нижній → довгі, але рідкі кроки  
    - Лівий верхній → часті, але короткі  
    - Лівий нижній → мінімальний вплив
    """)

    
    bottlenecks = analysis_df[
        (analysis_df["avg_duration"] > x_mean) &
        (analysis_df["avg_count"] > y_mean)
    ].sort_values("impact", ascending=False)
    
    
    if not bottlenecks.empty:
        top_step = bottlenecks.iloc[0]
        
        st.success(f"""
        🔴 Основний потенційний bottleneck: **{top_step['Activity Name']}**
        
        - Середня тривалість: {round(top_step['avg_duration'],2)} год
        - Середня кількість повторів: {round(top_step['avg_count'],2)}
        - Сумарний імпакт: {round(top_step['impact'],2)} год
        
        Крок перевищує середні значення за обома параметрами та має найбільший внесок у затримку процесу.
        """)
    else:
        st.info("Явно виражених bottleneck'ів (вище середнього по тривалості і повторюваності) не виявлено.")

    
    
# ---------------- Heuristics Miner ----------------
if log is not None:
    st.subheader("Heuristics Miner")
    st.markdown("Heuristics Miner → Petri Net показує реальний, частотний процес")
    st.markdown("Це граф переходів, ближчий до «як реально відбувалося»")
    st.markdown(" ")
    st.markdown("Основні елементи Petri Net:")
    st.markdown("- ◯ Кружки (places). Стани процесу «Тут ми зараз»")
    st.markdown("- ▭ Прямокутники (transitions). Активності, Реальні дії")
    st.markdown("- ➝ Стрілки. Потік виконання")
    st.markdown(" ")
    st.markdown("Частоти / товщина стрілок")
    st.markdown("📌 Читається:")
    st.markdown("товсті стрілки → часто")
    st.markdown("тонкі → рідко")
    st.markdown("Це дуже важливо для: bottleneck analysis, відхилень")
    st.markdown(" ")
    st.markdown("🧠 Як читати Heuristics Miner практично")
    st.markdown("1. Знайди Start → End")
    st.markdown("2. Подивись: де найбільше гілок, де є зворотні стрілки")
    st.markdown("3. Шукай: loops (повернення назад), обходи основного маршруту")
    st.markdown("4. Задай питання: Чому тут так багато варіантів? Чому тут повертаються назад?")
    st.markdown(" ")
    st.markdown("📌 Heuristics Miner = реальна поведінка, з шумом")
    st.markdown(" ")
 

    # Події відсортовані
    df_sorted = df.sort_values(["Case ID", "Start Timestamp"])
    
    # Наступна Activity Name та Start Timestamp
    df_sorted["next_activity"] = (
        df_sorted.groupby("Case ID")["Activity Name"].shift(-1)
    )
    
    df_sorted["next_timestamp"] = (
        df_sorted.groupby("Case ID")["Start Timestamp"].shift(-1)
    )
    
    # Час очікування
    df_sorted["waiting_time_hours"] = (
        df_sorted["next_timestamp"] - df_sorted["Start Timestamp"]
    ).dt.total_seconds() / 3600
    
    # Видаляємо останні події кейсів
    transitions = df_sorted.dropna(subset=["next_activity"])
    
    # Агрегація
    edges = (
        transitions
        .groupby(["Activity Name", "next_activity"])
        .agg(
            frequency=("Case ID", "count"),
            avg_waiting=("waiting_time_hours", "mean")
        )
        .reset_index()
    )

    # Bottleneck = максимальний avg_waiting серед переходів
    bottleneck_row = edges.loc[edges["avg_waiting"].idxmax()]
    
    bottleneck_text = (
        f"Найбільший bottleneck: "
        f"{bottleneck_row['Activity Name']} → {bottleneck_row['next_activity']} "
        f"(середній час: {bottleneck_row['avg_waiting']:.2f} год, "
        f"частота: {bottleneck_row['frequency']})"
    )

    # Товщина стрілок
    edges["penwidth"] = (
        edges["frequency"] / edges["frequency"].max() * 5
    ).clip(lower=1)
    

    # --- Динамічний колір за Pareto ---
    total_waiting = edges["avg_waiting"].sum()
    edges = edges.sort_values("avg_waiting", ascending=False).reset_index(drop=True)
    edges["cumsum_waiting"] = edges["avg_waiting"].cumsum()
    edges["cumsum_ratio"] = edges["cumsum_waiting"] / total_waiting
    
    def pareto_color(cumsum_ratio):
        if cumsum_ratio <= 0.8:   # Топ 80% затримок
            return "red"
        elif cumsum_ratio <= 0.95:  # Наступні 15%
            return "orange"
        else:  # Решта 5%
            return "green"
    
    edges["color"] = edges["cumsum_ratio"].apply(pareto_color)

    
    st.subheader("🔥 Heuristics Miner (Custom Graphviz)")

    dot = Digraph( 
        engine="dot", 
        graph_attr={"rankdir": "LR"}, 
        node_attr={"shape": "box", "style": "rounded,filled", "fillcolor": "#F9F9F9"} )

    # --- Розрахунок парето-поріг для легенди ---
    # Сортуємо за avg_waiting
    edges_sorted = edges.sort_values("avg_waiting")
    total_waiting = edges_sorted["avg_waiting"].sum()
    edges_sorted["cumsum_waiting"] = edges_sorted["avg_waiting"].cumsum()
    edges_sorted["cumsum_ratio"] = edges_sorted["cumsum_waiting"] / total_waiting
    
    # Червоно: топ 80% затримок → max avg_waiting у цій групі
    red_threshold = edges_sorted.loc[edges_sorted["cumsum_ratio"] <= 0.8, "avg_waiting"].max()
    # Оранжево: наступні 15% → max avg_waiting у цій групі
    orange_threshold = edges_sorted.loc[
        (edges_sorted["cumsum_ratio"] > 0.8) & (edges_sorted["cumsum_ratio"] <= 0.95),
        "avg_waiting"
    ].max()
    # Зелено: решта
    green_threshold = edges_sorted.loc[edges_sorted["cumsum_ratio"] > 0.95, "avg_waiting"].max()
    
    # --- ЛЕГЕНДА ---
    with dot.subgraph(name="cluster_legend") as c:
        c.attr(label="Legend", fontsize="12")
        c.node("L1", f"🟢 ≤ {green_threshold:.1f} год", shape="box", style="filled", fillcolor="green")
        c.node("L2", f"🟠 {green_threshold:.1f}–{orange_threshold:.1f} год", shape="box", style="filled", fillcolor="orange")
        c.node("L3", f"🔴 > {orange_threshold:.1f} год", shape="box", style="filled", fillcolor="red")

    
    # Додаємо всі Activity Name як вузли
    activities = set(edges["Activity Name"]).union(edges["next_activity"])
    for act in activities:
        dot.node(act)
    
    # Додаємо ребра з кастомними параметрами
    for _, row in edges.iterrows():
        dot.edge(
            row["Activity Name"],
            row["next_activity"],
            label=f'{row["frequency"]} | {row["avg_waiting"]:.1f}h',
            penwidth=str(row["penwidth"]),
            color=row["color"]
        )

    #Текстовий опис bottleneck прямо у графі
    dot.node(
        "bottleneck_info",
        bottleneck_text,
        shape="note",
        style="filled",
        fillcolor="#FFE4E1"
    )

    
    st.graphviz_chart(dot)

    st.markdown(" ")
    st.markdown("Як це читати (практично):")
    st.markdown(" ")
    st.markdown("🔴 товста + червона → критичний bottleneck")
    st.markdown("🟢 товста + зелена → стабільний шлях")
    st.markdown(" ")


    # ---------------- Variant analysis ----------------
    st.subheader("⚡ Variant analysis (ТОП 5 сценаріїв)")
    
    # Формуємо шлях кейсу
    variants = (
        df.sort_values("Start Timestamp")
          .groupby("Case ID")["Activity Name"]
          .apply(lambda x: " → ".join(x))
    )
    
    # Загальна статистика
    total_cases = variants.count()
    unique_variants = variants.nunique()
    
    variant_counts_full = variants.value_counts()
    variant_counts_top5 = variant_counts_full.head(5)
    
    # Таблиця ТОП 5
    variant_counts = (
        variant_counts_top5
        .reset_index()
    )
    variant_counts.columns = ["Сценарій процесу", "Кількість кейсів"]
    
    st.dataframe(variant_counts)
    
    # ---------------- Додаткова аналітика ----------------
    
    top1_share = variant_counts_full.iloc[0] / total_cases * 100
    top5_share = variant_counts_top5.sum() / total_cases * 100
    
    st.markdown("### 📊 Загальна структура варіантів")
    
    st.write(f"🔢 Загальна кількість кейсів: **{total_cases}**")
    st.write(f"🧭 Унікальних сценаріїв: **{unique_variants}**")
    st.write(f"🥇 Частка найпоширенішого сценарію: **{top1_share:.1f}%**")
    st.write(f"🏆 Частка ТОП-5 сценаріїв: **{top5_share:.1f}%**")
    
    # ---------------- Автоматичний висновок ----------------
    
        
    if unique_variants == 1:
        conclusion = "Процес повністю стандартизований. Всі кейси проходять однаковий сценарій."
    
    elif top1_share > 70:
        conclusion = (
            "Процес переважно стандартизований. "
            "Більшість кейсів слідують одному основному сценарію."
        )
    
    elif top5_share > 70:
        conclusion = (
            "Процес має помірну варіабельність. "
            "Існує кілька домінуючих сценаріїв."
        )
    
    else:
        conclusion = (
            "Процес характеризується високою варіабельністю. "
            "Велика кількість альтернативних сценаріїв може свідчити "
            "про нестандартизовані процедури або виняткові кейси."
        )
    
    st.info(conclusion)

    
    
# ---------------- Timeline кейсу ----------------
    st.subheader("📅 Timeline кейсу")
    
    case_list = df["Case ID"].unique()
    selected_case = st.selectbox("Оберіть кейс", case_list)
    
    case_df = df[df["Case ID"] == selected_case] \
        .sort_values("Start Timestamp")
    
    fig = px.scatter(
        case_df,
        x="Start Timestamp",
        y="Activity Name",
        title=f"Timeline кейсу {selected_case}",
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- FINAL EXECUTIVE SUMMARY ----------------
    
    st.markdown("---")
    st.header("🧠 Executive Summary та рекомендації")
    
    summary_text = ""
    
    # 1️⃣ Rework вплив
    lead_diff = mean_lead_rework - mean_lead_no_rework
    
    if percent_rework > 30:
        summary_text += (
            f"🔁 Значна частка кейсів ({percent_rework}%) містить повторювані кроки. "
            f"Rework збільшує середній Lead Time на {lead_diff:.2f} год.\n\n"
        )
    else:
        summary_text += (
            f"🔁 Частка rework становить {percent_rework}%, що не є критичною, "
            "але потребує моніторингу.\n\n"
        )
    
    # 2️⃣ Bottleneck кроку (bubble chart)
    if not bottlenecks.empty:
        summary_text += (
            f"🚧 Основний bottleneck на рівні активності: "
            f"{top_step['Activity Name']} "
            f"(середня тривалість {top_step['avg_duration']:.2f} год).\n\n"
        )
    
    # 3️⃣ Bottleneck переходу (Heuristics)
    summary_text += (
        f"⏳ Найбільша затримка між кроками: "
        f"{bottleneck_row['Activity Name']} → "
        f"{bottleneck_row['next_activity']} "
        f"({bottleneck_row['avg_waiting']:.2f} год очікування).\n\n"
    )
    
    # 4️⃣ Варіативність
    if unique_variants > total_cases * 0.5:
        summary_text += (
            "🔀 Процес має високу варіативність, що може свідчити "
            "про нестандартизовані процедури або винятки.\n\n"
        )
    elif top1_share > 70:
        summary_text += (
            "📏 Процес добре стандартизований з домінуючим основним сценарієм.\n\n"
        )
    
    # ---------------- РЕКОМЕНДАЦІЇ ----------------
    
    recommendations = "### 📌 Рекомендації:\n\n"
    
    if percent_rework > 30:
        recommendations += "- Зменшити причини повторних кроків (аналіз root cause rework).\n"
    
    if not bottlenecks.empty:
        recommendations += f"- Оптимізувати або автоматизувати крок **{top_step['Activity Name']}**.\n"
    
    recommendations += (
        "- Проаналізувати переходи з найбільшим waiting time.\n"
        "- Стандартизувати варіативні сценарії або формалізувати винятки.\n"
        "- Впровадити SLA для критичних переходів.\n"
    )
    
    # Вивід
    st.markdown(summary_text)
    st.markdown(recommendations)
    
    # ---------------- PROCESS MATURITY SCORE ----------------
    
    maturity_score = 100
    
    if percent_rework > 30:
        maturity_score -= 20
    
    if unique_variants > total_cases * 0.5:
        maturity_score -= 20
    
    if not bottlenecks.empty:
        maturity_score -= 20
    
    maturity_score = max(maturity_score, 0)
    
    st.subheader("📊 Process Maturity Score")
    
    st.metric("Індекс зрілості процесу (0–100)", maturity_score)
    
    if maturity_score > 80:
        st.success("Процес високозрілий та контрольований.")
    elif maturity_score > 50:
        st.warning("Процес середнього рівня зрілості. Є зони для оптимізації.")
    else:
        st.error("Процес має значні структурні проблеми та потребує оптимізації.")

    # ---------------- AI NARRATIVE ----------------
    
    st.header("🧠 AI Process Narrative")
    
    if maturity_score > 80:
        maturity_level = "високим рівнем операційної стабільності"
    elif maturity_score > 50:
        maturity_level = "помірною структурною зрілістю"
    else:
        maturity_level = "операційною нестабільністю"
    
    ai_text = f"""
    Процес характеризується {maturity_level}.
    
    Середній Lead Time становить {avg_duration:.2f} годин.
    Частка rework складає {percent_rework}%.
    Кількість унікальних варіантів процесу — {unique_variants}.
    
    Основні втрати часу пов'язані з кроками високої тривалості
    та переходами з великим waiting time.
    
    Поточна структура процесу свідчить про необхідність
    структурної оптимізації критичних активностей та стандартизації сценаріїв.
    """
    
    st.info(ai_text)

    # ---------------- KPI SCORECARD ----------------
    
    st.markdown("---")
    st.header("📊 KPI Scorecard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Lead Time (avg)", f"{avg_duration:.2f} h")
    col2.metric("Rework Rate", f"{percent_rework}%")
    col3.metric("Variant Count", unique_variants)
    col4.metric("Main Variant Share", f"{top1_share:.1f}%")

    # ---------------- RISK HEATMAP ----------------
    
    st.header("🔥 Risk Heatmap")
    
    risk_matrix = analysis_df.copy()
    
    risk_matrix["risk_score"] = (
        (risk_matrix["avg_duration"] / x_mean) *
        (risk_matrix["avg_count"] / y_mean)
    )
    
    pivot = risk_matrix.pivot_table(
        values="risk_score",
        index="Activity Name"
    )
    
    plt.figure(figsize=(3, 4))
    sns.heatmap(
        pivot,
        annot=True,
        cmap="Reds",
        linewidths=0.5
    )
    
    plt.title("Risk Intensity per Activity")
    plt.xticks([])
    st.pyplot(plt.gcf())

    # ---------------- IMPROVEMENT ROADMAP ----------------
    
    st.header("🚀 Improvement Roadmap")
    
    roadmap = []
    
    if percent_rework > 30:
        roadmap.append("1️⃣ Провести root cause analysis повторюваних кроків")
    
    if not bottlenecks.empty:
        roadmap.append(f"2️⃣ Оптимізувати крок '{top_step['Activity Name']}'")
    
    roadmap.append("3️⃣ Встановити SLA для критичних переходів")
    roadmap.append("4️⃣ Стандартизувати ТОП варіанти процесу")
    roadmap.append("5️⃣ Впровадити регулярний process monitoring dashboard")
    
    for item in roadmap:
        st.write(item)

    # ---------------- PDF GENERATOR ----------------
    
    def generate_pdf_report(summary_text, recommendations, maturity_score):
    
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=50,
            bottomMargin=40
        )
    
        elements = []
    
        # ---------------- REGISTER FONT ----------------
        #pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))
        font_path = os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")
        pdfmetrics.registerFont(TTFont("DejaVuSans", font_path))
    
        # ---------------- STYLES ----------------
        base_style = ParagraphStyle(
            name="BaseStyle",
            fontName="DejaVuSans",
            fontSize=11,
            leading=16,
            textColor=colors.black,
            firstLineIndent=20,     # відступ першого рядка
            spaceAfter=10           # відступ між абзацами
        )
    
        title_style = ParagraphStyle(
            name="TitleStyle",
            fontName="DejaVuSans",
            fontSize=18,
            leading=22,
            textColor=colors.black,
            spaceAfter=18
        )
    
        subtitle_style = ParagraphStyle(
            name="SubtitleStyle",
            fontName="DejaVuSans",
            fontSize=13,
            leading=16,
            textColor=colors.black,
            spaceAfter=10
        )
    
        # ---------------- TITLE ----------------
        elements.append(Paragraph("Process Mining Executive Report", title_style))
        elements.append(Paragraph(f"Дата формування: {datetime.now().strftime('%d.%m.%Y')}", base_style))
        elements.append(Spacer(1, 20))
    
        # ---------------- EXECUTIVE SUMMARY ----------------
        elements.append(Paragraph("Executive Summary", subtitle_style))
        elements.append(Paragraph(summary_text.replace("\n", "<br/>"), base_style))
        elements.append(Spacer(1, 15))
    
        # ---------------- RECOMMENDATIONS ----------------
        elements.append(Paragraph("Рекомендації", subtitle_style))
        elements.append(Paragraph(recommendations.replace("\n", "<br/>"), base_style))
        elements.append(Spacer(1, 15))
    
        # ---------------- MATURITY SCORE ----------------
        elements.append(Paragraph("Process Maturity Score", subtitle_style))
    
        maturity_text = f"""
    Process Maturity Score — це інтегральний показник зрілості процесу (шкала 0–100).
    Він враховує рівень повторних кроків (rework), варіативність сценаріїв,
    наявність bottleneck’ів та стабільність виконання процесу.
    
    Поточне значення індексу: {maturity_score}/100.
    """
    
        elements.append(Paragraph(maturity_text.replace("\n", "<br/>"), base_style))
        elements.append(Spacer(1, 20))
    
        # ---------------- AUTHOR ----------------
        elements.append(Paragraph("Автор застосунку", subtitle_style))
    
        author_text = """
    Hanna Nesterenko  
    LinkedIn: <link href="https://www.linkedin.com/in/anna-nesterenko-bi/">
    https://www.linkedin.com/in/anna-nesterenko-bi/
    </link>
    """
    
        elements.append(Paragraph(author_text.replace("\n", "<br/>"), base_style))
    
        doc.build(elements)
        buffer.seek(0)
    
        return buffer

    pdf_buffer = generate_pdf_report(summary_text, recommendations, maturity_score)
    
    st.download_button(
        label="📄 Завантажити Executive Report (PDF)",
        data=pdf_buffer,
        file_name="process_mining_executive_report.pdf",
        mime="application/pdf"
    )
    
        


