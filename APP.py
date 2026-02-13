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
from io import BytesIO

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


# ---------------- UI ----------------
st.set_page_config(page_title="Process Mining (Excel)", layout="wide")
st.title("🧩 Process Mining App")
# ---------------- Авторство ----------------
st.markdown("---")
st.markdown("© 2026 Hanna Nesterenko | [LinkedIn](https://www.linkedin.com/in/anna-nesterenko-bi/)")
st.markdown("---")
st.markdown("Завантажте Excel-файл з подіями для аналізу процесів")
st.markdown("Файл має міститі обов'язкові поля (кожен рядок = подія/крок (event)):")
st.markdown("- case_id — унікальний номер або назва кейсу")
st.markdown("- activity — назва події/кроку")
st.markdown("- timestamp — дата й час початку події/кроку")

# ---------------- Upload Excel ----------------
uploaded_file = st.file_uploader("Завантажте Excel лог", type=["xlsx"])

log = None
df = None

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    required_cols = {"case_id", "activity", "timestamp"}
    if not required_cols.issubset(df.columns):
        st.error("Excel має містити колонки: case_id, activity, timestamp")
        st.stop()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = dataframe_utils.convert_timestamp_columns_in_df(df)

    st.success("Excel успішно завантажено")
    st.dataframe(df.head(5))

    # ---------------- Convert to EventLog ----------------
    log = EventLog()

    for case_id, group in df.groupby("case_id"):
        trace = Trace()
        trace.attributes["concept:name"] = str(case_id)

        for _, row in group.sort_values("timestamp").iterrows():
            event = Event()
            event["concept:name"] = row["activity"]
            event["time:timestamp"] = row["timestamp"]
            trace.append(event)

        log.append(trace)

    #st.info(f"Кількість кейсів: {len(log)}")

    
# ---------------- Base analytics ----------------
    st.subheader("📊 Загальна статистика логів")

    # --- Кількість кейсів ---
    num_cases = df["case_id"].nunique()
    
    # --- Період дослідження ---
    start_period = df["timestamp"].min()
    end_period = df["timestamp"].max()
    
    # --- Тривалість кейсів ---
    case_times = (
        df.groupby("case_id")["timestamp"]
        .agg(start="min", end="max")
        .reset_index()
    )
    case_times["duration_hours"] = (
        case_times["end"] - case_times["start"]
    ).dt.total_seconds() / 3600
    
    avg_duration = case_times["duration_hours"].mean()
    median_duration = case_times["duration_hours"].median()
    
    # --- Кількість activity на кейс ---
    activities_per_case = (
        df.groupby("case_id")["activity"]
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
        st.metric("Сер. кількість activity на кейс",
              round(avg_activities, 1))

    
    most_common_start = (
        df.sort_values("timestamp")
          .groupby("case_id")
          .head(1)["activity"]
          .value_counts()
          .idxmax()
    )
    
    most_common_end = (
        df.sort_values("timestamp")
          .groupby("case_id")
          .tail(1)["activity"]
          .value_counts()
          .idxmax()
    )

    # Кількість повторів activity в межах кейсу
    activity_repeats = (
        df.groupby(["case_id", "activity"])
          .size()
          .reset_index(name="count")
    )
    
    # Беремо тільки ті, що повторювались
    repeated_steps = activity_repeats[
        activity_repeats["count"] > 1
    ]
    
    top_rework = (
        repeated_steps.groupby("activity")["count"]
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
        df.groupby("case_id")["timestamp"]
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
        df.sort_values("timestamp")
          .groupby("case_id")
          .tail(1)["activity"]
    )
    
    top_end_activities = last_activities.value_counts().head(10)
    
    st.write("ТОП кроків завершення:")
    st.dataframe(top_end_activities.reset_index()
                 .rename(columns={"index": "activity",
                                  "activity": "кількість кейсів"}))


    # ---------------- Rework ----------------
    # Обчислення кейсів з повтореннями 
    activity_counts = (
        df.groupby("case_id")["activity"]
          .value_counts()
          .reset_index(name="count")
    )
    
    # Визначаємо кейси, де якась активність повторюється більше 1 разу
    cases_with_rework_list = activity_counts.loc[activity_counts["count"] > 1, "case_id"].unique()
    
    st.subheader("🔁 Повторювані кроки (rework)")
    
    # ТОП кроків з повтореннями
    top_rework = activity_counts.groupby("activity")["count"].sum().sort_values(ascending=False).head(10)
    st.write("ТОП кроків з повтореннями:")
    st.dataframe(top_rework.reset_index().rename(columns={"count": "кількість повторів"}))
    
    # Аналітичний висновок по кейсам з повтореннями
    total_rework_cases = len(cases_with_rework_list)
    total_cases = df["case_id"].nunique()
    percent_rework = round((total_rework_cases / total_cases) * 100, 2)
    st.markdown(
        f"В нашій вибірці {total_rework_cases} кейсів ({percent_rework}%) містять повторювані кроки. "
        "Це вказує на наявність rework у процесі, який уповільнює його виконання та підвищує варіабельність тривалості кейсів."
    )
    
    # ---------------- Графік Lead Time ----------------
    st.markdown("### 📈 Розподіл Lead Time: кейси з rework vs без")
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Групуємо по кейсу і обчислюємо Lead Time (години)
    lead_time_per_case = (
        df.groupby("case_id")["timestamp"]
          .agg(lead_time=lambda x: (x.max() - x.min()).total_seconds() / 3600)
          .reset_index()
    )
    
    # Додаємо колонку Rework
    lead_time_per_case["rework"] = lead_time_per_case["case_id"].isin(cases_with_rework_list)
    lead_time_per_case["rework_label"] = lead_time_per_case["rework"].map({True: "З повтореннями", False: "Без повторень"})
    
    # Фігура
    plt.figure(figsize=(5,3))
    
    # Гістограма + Boxplot
    #sns.histplot(
        #data=lead_time_per_case,
        #x="lead_time",
        #hue="rework_label",
        #bins=20,
        #kde=True,
        #palette={"З повтореннями": "red", "Без повторень": "green"},
        #alpha=0.6
    #)
    
    sns.boxplot(
        data=lead_time_per_case,
        x="lead_time",
        y="rework_label",
        palette={"З повтореннями": "red", "Без повторень": "green"},
        width=0.3,
        fliersize=3
    )
    
    plt.xlabel("Lead Time (год)")
    plt.ylabel("")
    plt.title("Розподіл тривалості кейсів з Rework та без")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    
    # ---------------- Середні показники ----------------

    # Припустимо, Waiting Time = Lead Time мінус суму тривалостей активностей
    # Спрощено, якщо у немає фактичної тривалості активностей, можна просто як дельту між кроками:
    
    waiting_time_per_case = (
        df.groupby("case_id")["timestamp"]
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
    df_sorted = df.sort_values(["case_id", "timestamp"])
    
    # Наступна activity та timestamp
    df_sorted["next_activity"] = (
        df_sorted.groupby("case_id")["activity"].shift(-1)
    )
    
    df_sorted["next_timestamp"] = (
        df_sorted.groupby("case_id")["timestamp"].shift(-1)
    )
    
    # Час очікування
    df_sorted["waiting_time_hours"] = (
        df_sorted["next_timestamp"] - df_sorted["timestamp"]
    ).dt.total_seconds() / 3600
    
    # Видаляємо останні події кейсів
    transitions = df_sorted.dropna(subset=["next_activity"])
    
    # Агрегація
    edges = (
        transitions
        .groupby(["activity", "next_activity"])
        .agg(
            frequency=("case_id", "count"),
            avg_waiting=("waiting_time_hours", "mean")
        )
        .reset_index()
    )

    # Bottleneck = максимальний avg_waiting серед переходів
    bottleneck_row = edges.loc[edges["avg_waiting"].idxmax()]
    
    bottleneck_text = (
        f"Найбільший bottleneck: "
        f"{bottleneck_row['activity']} → {bottleneck_row['next_activity']} "
        f"(середній час: {bottleneck_row['avg_waiting']:.2f} год, "
        f"частота: {bottleneck_row['frequency']})"
    )

    # Товщина стрілок
    edges["penwidth"] = (
        edges["frequency"] / edges["frequency"].max() * 5
    ).clip(lower=1)
    
    # Колір за waiting time
    def waiting_to_color(hours):
        if hours < 1:
            return "green"
        elif hours < 4:
            return "orange"
        else:
            return "red"
    
    edges["color"] = edges["avg_waiting"].apply(waiting_to_color)
    
    st.subheader("🔥 Heuristics Miner (Custom Graphviz)")
    
    dot = Digraph(
        engine="dot",
        graph_attr={"rankdir": "LR"},
        node_attr={"shape": "box", "style": "rounded,filled", "fillcolor": "#F9F9F9"}
    )

    # --- ЛЕГЕНДА ---
    with dot.subgraph(name="cluster_legend") as c:
        c.attr(label="Legend", fontsize="12")
        c.node("L1", "🟢 < 1 год", shape="box", style="filled", fillcolor="green")
        c.node("L2", "🟠 1–4 год", shape="box", style="filled", fillcolor="orange")
        c.node("L3", "🔴 > 4 год", shape="box", style="filled", fillcolor="red")
    
    # Додаємо всі activity як вузли
    activities = set(edges["activity"]).union(edges["next_activity"])
    for act in activities:
        dot.node(act)
    
    # Додаємо ребра з кастомними параметрами
    for _, row in edges.iterrows():
        dot.edge(
            row["activity"],
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

    
    
    st.markdown(" ")
    st.markdown(" ")


    # ---------------- Variant analysis ----------------
    st.subheader("⚡ Variant analysis (ТОП 5 сценаріїв)")
    
    # Формуємо шлях кейсу
    variants = (
        df.sort_values("timestamp")
          .groupby("case_id")["activity"]
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
    
    case_list = df["case_id"].unique()
    selected_case = st.selectbox("Оберіть кейс", case_list)
    
    case_df = df[df["case_id"] == selected_case] \
        .sort_values("timestamp")
    
    fig = px.scatter(
        case_df,
        x="timestamp",
        y="activity",
        title=f"Timeline кейсу {selected_case}",
    )
    
    st.plotly_chart(fig, use_container_width=True)

    


