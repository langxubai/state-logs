# 使用官方 Python 瘦身镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量，防止 Python 生成 .pyc 文件并且让控制台输出不被缓冲
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 更新 apt 并安装一些必要的系统依赖（如果有需要的话）
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements 文件并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 将当前目录下的所有文件复制到容器的工作目录
COPY . .

# 暴露 Streamlit 的默认运行端口
EXPOSE 8501

# 添加健康检查（可选，确保容器正常运行）
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 配置运行 Streamlit 时的默认命令
ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
