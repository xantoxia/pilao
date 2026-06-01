import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import streamlit as st
from matplotlib import font_manager
import os
from openai import OpenAI
import base64
import requests
import datetime
import io
import pytz


GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_USERNAME = 'xantoxia'
GITHUB_REPO = 'blank-app-1'
GITHUB_BRANCH = 'main'
FILE_PATH = 'fatigue_data.csv'


def get_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return ""
    except UnicodeDecodeError:
        st.error("文件编码错误，无法解码文件。")
        return None


def get_file_sha(file_path):
    url = f'https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/{file_path}'
    headers = {'Authorization': f'token {GITHUB_TOKEN}'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()['sha']
    else:
        st.warning(f"无法从 GitHub 获取文件: {response.json()}")
        return None


def save_to_csv(input_data, result, body_fatigue, cognitive_fatigue, emotional_fatigue):
    body_fatigue_score = calculate_score(body_fatigue)
    cognitive_fatigue_score = calculate_score(cognitive_fatigue)
    emotional_fatigue_score = calculate_score(emotional_fatigue)

    tz = pytz.timezone('Asia/Shanghai')
    timestamp = datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')

    data = {
        "颈部前屈": int(input_data["颈部前屈"].values[0]),
        "颈部后仰": int(input_data["颈部后仰"].values[0]),
        "肩部上举范围": int(input_data["肩部上举范围"].values[0]),
        "肩部前伸范围": int(input_data["肩部前伸范围"].values[0]),
        "肘部屈伸": int(input_data["肘部屈伸"].values[0]),
        "手腕背伸": int(input_data["手腕背伸"].values[0]),
        "手腕桡偏/尺偏": int(input_data["手腕桡偏/尺偏"].values[0]),
        "背部屈曲范围": int(input_data["背部屈曲范围"].values[0]),
        "持续时间": int(input_data["持续时间"].values[0]),
        "重复频率": int(input_data["重复频率"].values[0]),
        "fatigue_result": result,
        "body_fatigue_score": body_fatigue_score,
        "cognitive_fatigue_score": cognitive_fatigue_score,
        "emotional_fatigue_score": emotional_fatigue_score,
        "timestamp": timestamp
    }

    df = pd.DataFrame([data])

    if os.path.exists(FILE_PATH):
        existing_content = get_file_content(FILE_PATH)
        if existing_content and existing_content.strip():
            existing_df = pd.read_csv(io.StringIO(existing_content))
        else:
            existing_df = pd.DataFrame(columns=df.columns)
    else:
        existing_df = pd.DataFrame(columns=df.columns)

    updated_df = pd.concat([existing_df, df], ignore_index=True)
    updated_df.to_csv(FILE_PATH, index=False)


def upload_to_github(file_path):
    sha_value = get_file_sha(file_path)

    with open(file_path, 'rb') as file:
        content = base64.b64encode(file.read()).decode()

    url = f'https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/{file_path}'

    data = {
        "message": "Add new fatigue data with timestamp",
        "branch": GITHUB_BRANCH,
        "content": content,
    }

    if sha_value:
        data["sha"] = sha_value

    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json',
    }

    response = requests.put(url, json=data, headers=headers)

    if response.status_code not in [200, 201]:
        st.error(f"Failed to upload CSV file to GitHub: {response.json()}")


def calculate_score(answer):
    if answer == '请选择':
        return 0
    elif answer == '完全没有':
        return 1
    elif answer == '偶尔':
        return 2
    elif answer == '经常':
        return 3
    else:
        return 4


# =========================
# AI部分（已修复核心问题）
# =========================

def call_ark_api(client, messages):
    try:
        ark_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

        completion = client.chat.completions.create(
            model="Pro/deepseek-ai/DeepSeek-V3.2",
            messages=ark_messages,
            stream=True
        )

        for chunk in completion:
            delta = getattr(chunk.choices[0].delta, "content", None)
            if delta:
                yield delta

    except Exception as e:
        st.error(f"调用 AI API 出错：{e}")
        yield ""


# =========================
# ⭐关键修复1：初始化状态
# =========================
if "ai_analysis_result" not in st.session_state:
    st.session_state.ai_analysis_result = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "client" not in st.session_state:
    st.session_state.client = None

if "API_KEY" not in st.session_state:
    st.session_state.API_KEY = None

if "api_key_entered" not in st.session_state:
    st.session_state.api_key_entered = False


# =========================
# 原模型部分（完全不动）
# =========================
font_path = "SourceHanSansCN-Normal.otf"

if os.path.exists(font_path):
    font_prop = font_manager.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False


file_path = 'corrected_fatigue_simulation_data_Chinese.csv'
data = pd.read_csv(file_path, encoding='gbk')

X = data.drop(columns=["疲劳等级"])
y = data["疲劳等级"]

X.columns = X.columns.str.replace(' ', '_')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


def fatigue_prediction(input_data):
    prediction = model.predict(input_data)
    return ["低疲劳状态", "中疲劳状态", "高疲劳状态"][prediction[0]]


# =========================
# AI按钮（最小修改）
# =========================
if st.button("开始 AI 分析"):

    st.subheader("AI 分析")
    st.info("生成潜在人因危害分析及改善建议：")

    API_KEY = st.secrets["API_KEY"]

    # ⭐关键修复2：只初始化一次 client
    if API_KEY and st.session_state.client is None:
        st.session_state.client = OpenAI(
            api_key=API_KEY,
            base_url="https://api.siliconflow.cn/v1"
        )
        st.session_state.api_key_entered = True
        st.session_state.API_KEY = API_KEY

    if st.session_state.client and st.session_state.result:

        ai_input = f"""
用户状态：
身体疲劳={body_fatigue}
睡眠影响={cognitive_fatigue}
肌肉酸痛={emotional_fatigue}

动作数据：
颈部前屈{neck_flexion}°
颈部后仰{neck_extension}°
肩部上举{shoulder_elevation}°
肩部前伸{shoulder_forward}°
肘部{elbow_flexion}°
手腕背伸{wrist_extension}°
手腕偏移{wrist_deviation}°
背部屈曲{back_flexion}°

请做人因风险分析并给出优先改善建议。
"""

        st.session_state.messages = [
            {"role": "system", "content": "你是人因工程专家。"},
            {"role": "user", "content": ai_input}
        ]

        with st.spinner("AI分析中..."):
            response = ""
            for chunk in call_ark_api(st.session_state.client, st.session_state.messages):
                response += chunk

            if response:
                st.session_state.ai_analysis_result = response
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                st.success("分析完成")
            else:
                st.error("AI返回为空")
