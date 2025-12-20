# ==============================
#  构建阶段：仅安装 Python 依赖
# ==============================
FROM python:3.13-slim-bookworm AS builder

# 1. 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app/lunabot_nonebot

# 2. 设置 uv 缓存目录以加速构建
ENV UV_CACHE_DIR=/root/.cache/uv

# 3. 复制依赖描述文件
# 优先复制 pyproject.toml 和 uv.lock 以利用 Docker 层缓存
COPY pyproject.toml uv.lock ./

# 4. 安装依赖
# --frozen: 强制使用 lock 文件中的版本
# --no-install-project: 在复制源码前先安装依赖，避免源码变动导致重新下载依赖
RUN --mount=type=cache,target=$UV_CACHE_DIR \
    uv sync --frozen --no-install-project --no-dev

# ==============================
#  运行阶段
# ==============================
FROM python:3.13-slim-bookworm AS runtime

# 设置工作目录
WORKDIR /app/lunabot_nonebot

# 从构建阶段复制虚拟环境
COPY --from=builder /app/lunabot_nonebot/.venv /app/lunabot_nonebot/.venv

# 设置时区，配置环境变量，确保优先使用虚拟环境中的 Python 和 Bin
ENV TZ=Asia/Shanghai \
    PATH="/app/lunabot_nonebot/.venv/bin:$PATH"

# 安装 opencv 所需库
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    
# 安装playwright
RUN playwright install --only-shell --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 8383 8486

# 挂载数据目录
VOLUME ["/app/lunabot_nonebot/data", "/app/lunabot_nonebot/config"]
# 启动 NoneBot
CMD ["python", "bot.py"]
