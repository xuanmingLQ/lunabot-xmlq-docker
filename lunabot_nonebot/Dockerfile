# ==============================
#  构建阶段：仅安装 Python 依赖
# ==============================
FROM python:3.13-slim-bookworm AS builder

WORKDIR /app/lunabot_nonebot

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 包
RUN pip --no-cache-dir -r requirements.txt

# ==============================
#  运行阶段
# ==============================
FROM python:3.13-slim-bookworm AS runtime

# 设置时区
ENV TZ=Asia/Shanghai

WORKDIR /app/lunabot_nonebot

# 安装 运行所需库
RUN apt-get update && apt-get install -y --no-install-recommends \
    -o Acquire::http::Timeout="120" \
    -o Acquire::http::Max-Retries="5" \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    fonts-noto-color-emoji \
    # 设置时区
    tzdata \
    openntpd \
    # 下载中文字体
    fontconfig \
    ttf-wqy-zenhei \
    && ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
    
# 复制 Python 依赖
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
# Playwright 安装（仅在运行镜像中执行）
RUN  playwright install chromium

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 8383

# 挂载数据目录
VOLUME ["/app/lunabot_nonebot/data", "/app/lunabot_nonebot/config", "/app/lunabot_nonebot/.env"]
# 启动 NoneBot
CMD ["python", "bot.py"]
