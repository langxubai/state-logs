import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
import os

# ==========================================
# 物理模型参数配置
# ==========================================
TAU_S = 2.0   # 状态弛豫时间常数（天）
TAU_B = 14.0  # 基线演化时间常数（天）
DATA_FILE = "state_logs.csv"

# ==========================================
# 数据处理与存储函数
# ==========================================
def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df.sort_values('Timestamp').reset_index(drop=True)
    else:
        return pd.DataFrame(columns=['Timestamp', 'Input', 'Note'])

def save_data(timestamp, value, note):
    df = load_data()
    new_row = pd.DataFrame({'Timestamp': [timestamp], 'Input': [value], 'Note': [note]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    return df

# ==========================================
# 核心动力学演化与插值算法
# ==========================================
def calculate_dynamics(df):
    if df.empty:
        df_augmented = df.copy()
        df_augmented['实际改变量'] = []
        return pd.DataFrame(), pd.DataFrame(), df_augmented

    # 存储用于画连续曲线的密集数据点
    plot_times = []
    plot_S = []
    plot_B = []
    
    # 存储事件触发瞬间的数据点（用于画散点和Tooltip）
    event_times = []
    event_S = []
    event_Notes = []
    
    actual_jumps = [] # 记录每次事件的实际改变量

    # 初始状态
    t_current = df['Timestamp'].iloc[0]
    S_current = 0.0
    B_current = 0.0

    for index, row in df.iterrows():
        t_event = row['Timestamp']
        I = row['Input']
        note = row['Note']
        
        dt_total = (t_event - t_current).total_seconds() / 86400.0 # 转换为天
        
        # 1. 如果两次记录之间有时间差，生成插值点画平滑的指数演化曲线
        if dt_total > 0:
            num_steps = max(int(dt_total * 10), 2) # 每天插值10个点
            t_steps = np.linspace(0, dt_total, num_steps)
            
            for dt in t_steps[1:]: # 跳过第0个点（已在上一轮记录）
                S_t = B_current + (S_current - B_current) * np.exp(-dt / TAU_S)
                B_t = B_current + (S_current - B_current) * (1 - np.exp(-dt / TAU_B))
                
                plot_times.append(t_current + pd.Timedelta(days=dt))
                plot_S.append(S_t)
                plot_B.append(B_t)
                
            # 更新到事件发生前一瞬间的物理状态
            S_current = B_current + (S_current - B_current) * np.exp(-dt_total / TAU_S)
            B_current = B_current + (S_current - B_current) * (1 - np.exp(-dt_total / TAU_B))
            t_current = t_event

        # 2. 事件触发瞬间，状态波函数坍缩 (跃迁)
        intended_S = B_current + I
        delta_S = intended_S - S_current
        
        # 引入“感知失灵/心理惯性”机制 (Psychological Inertia)
        # 当情绪预期跃迁幅度 >= 2 时，人往往会有“钝感”，实际跃迁幅度被打折
        if abs(delta_S) >= 2.0:
            # > 2.0 的部分进行阻尼衰减（这里先暂定设为 50% 的效果）
            actual_delta = 2.0 + (abs(delta_S) - 2.0) * 0.5
            actual_jump = np.sign(delta_S) * actual_delta
            S_current = S_current + actual_jump
        else:
            actual_jump = delta_S
            S_current = intended_S
            
        actual_jumps.append(round(actual_jump, 2))
        
        # 记录事件点
        event_times.append(t_event)
        event_S.append(S_current)
        event_Notes.append(f"{note} (输入: {I:>+})")
        
        # 记录到连续曲线中
        plot_times.append(t_current)
        plot_S.append(S_current)
        plot_B.append(B_current)
        
    # 3. 把最后一次记录演化到现在（如果最后一次记录在过去）
    now = pd.Timestamp.now()
    dt_to_now = (now - t_current).total_seconds() / 86400.0
    if dt_to_now > 0:
        num_steps = max(int(dt_to_now * 10), 2)
        t_steps = np.linspace(0, dt_to_now, num_steps)
        for dt in t_steps[1:]:
            S_t = B_current + (S_current - B_current) * np.exp(-dt / TAU_S)
            B_t = B_current + (S_current - B_current) * (1 - np.exp(-dt / TAU_B))
            plot_times.append(t_current + pd.Timedelta(days=dt))
            plot_S.append(S_t)
            plot_B.append(B_t)

    df_plot = pd.DataFrame({'Time': plot_times, 'State': plot_S, 'Baseline': plot_B})
    df_events = pd.DataFrame({'Time': event_times, 'State': event_S, 'Note': event_Notes})
    
    # 将实际改变量附加到原始 df 返回
    df_augmented = df.copy()
    df_augmented['实际改变量'] = actual_jumps
    return df_plot, df_events, df_augmented

# ==========================================
# Streamlit 前端界面
# ==========================================
st.set_page_config(page_title="个人状态动力学系统", layout="wide")
st.title("🌊 个人状态双时间尺度演化模型")

df = load_data()

# 侧边栏：数据输入区
with st.sidebar:
    st.header("📝 记录当前状态")
    
    # 获取东八区当前时间（兼容本地和云端部署的 UTC 时差问题）
    tz_zh = timezone(timedelta(hours=8))
    now = datetime.now(tz_zh)
    
    use_now = st.toggle("🕒 同步最新时间", value=True, help="关闭此项即可手动修改时间，用于补填历史状态")
    
    if use_now:
        # 当开启同步时，框体禁用，防止误触，保证获取到你点击保存那一刻的最新时间
        st.date_input("日期", value=now.date(), disabled=True)
        st.time_input("时间", value=now.time(), disabled=True)
        event_timestamp = pd.Timestamp(now).tz_localize(None) # 剥离时区信息以兼容原有存储
    else:
        record_date = st.date_input("日期", value=now.date())
        record_time = st.time_input("时间", value=now.time())
        event_timestamp = pd.Timestamp(datetime.combine(record_date, record_time))
    
    input_value = st.select_slider(
        "你的直觉感受是？",
        options=[-2, -1, 0, 1, 2],
        value=0,
        format_func=lambda x: {-2: "💥 极差 (-2)", -1: "😫 糟糕 (-1)", 0: "😐 平稳 (0)", 1: "😊 不错 (+1)", 2: "🤩 极好 (+2)"}[x]
    )
    
    note = st.text_input("发生了什么？(可选)", placeholder="例如：推导出了一个漂亮的公式...")
    
    if st.button("💾 记录状态", type="primary"):
        save_data(event_timestamp, input_value, note)
        st.success("记录成功！状态已更新。")
        st.rerun()

# 主界面：可视化区
df_plot, df_events, df_augmented = calculate_dynamics(df)

if not df_plot.empty:
    st.markdown("### 📊 实时状态看板")
    col1, col2, col3 = st.columns(3)
    latest_state = df_plot['State'].iloc[-1]
    latest_baseline = df_plot['Baseline'].iloc[-1]
    gap = latest_state - latest_baseline
    
    col1.metric("当前瞬时状态 (State)", f"{latest_state:.2f}")
    col2.metric("当前常态基线 (Baseline)", f"{latest_baseline:.2f}")
    col3.metric("状态偏离度 (Gap)", f"{gap:+.2f}")

    fig = go.Figure()

    # 1. 画 14天平均基线 (虚线)
    fig.add_trace(go.Scatter(
        x=df_plot['Time'], y=df_plot['Baseline'],
        mode='lines',
        name='常态基线 (14天慢弛豫)',
        line=dict(color='orange', width=2, dash='dash')
    ))

    # 2. 画 实际状态线 (实线)
    fig.add_trace(go.Scatter(
        x=df_plot['Time'], y=df_plot['State'],
        mode='lines',
        name='当前状态 (2天快弛豫)',
        line=dict(color='#1f77b4', width=3, shape='spline'),
        fill='tonexty', # 填充与基线之间的面积，视觉效果极佳
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))

    # 3. 画 打卡事件的散点 (带悬浮提示)
    fig.add_trace(go.Scatter(
        x=df_events['Time'], y=df_events['State'],
        mode='markers',
        name='记录锚点',
        marker=dict(color='red', size=8, symbol='circle', line=dict(color='white', width=1)),
        text=df_events['Note'],
        hoverinfo='text+y'
    ))

    fig.update_layout(
        height=500,
        hovermode="x unified",
        xaxis_title="时间",
        yaxis_title="绝对状态分数",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # 显示近期记录表格
    with st.expander("📝 数据管理与修正 (直接在表格中修改或删除)"):
        edited_df = st.data_editor(
            df_augmented.sort_values('Timestamp', ascending=False),
            disabled=["实际改变量"],
            num_rows="dynamic",
            use_container_width=True,
            key="data_editor"
        )
        if st.button("💾 保存修改的表格"):
            final_df = edited_df.sort_values('Timestamp').reset_index(drop=True)
            if "实际改变量" in final_df.columns:
                final_df = final_df.drop(columns=["实际改变量"])
            final_df.to_csv(DATA_FILE, index=False)
            st.success("数据修改已保存！")
            st.rerun()
        
else:
    st.info("👈 目前还没有数据，请在左侧侧边栏记录你的第一次状态！")