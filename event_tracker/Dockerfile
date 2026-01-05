FROM python:3.13-slim-bookworm

# 1. 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 设置工作目录
WORKDIR /app/lunabot_event_tracker

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

# 5. 复制应用程序代码
COPY . .

# 6. 安装项目本身
RUN --mount=type=cache,target=$UV_CACHE_DIR \
    uv sync --frozen --no-dev

# 配置环境变量
# 确保虚拟环境中的 python 路径在最前面
ENV PATH="/app/lunabot_event_tracker/.venv/bin:$PATH" \
    TZ=Asia/Shanghai
# 设置时区
RUN echo "deb https://mirrors.aliyun.com/debian/ bookworm main" > /etc/apt/sources.list \
    && echo "deb https://mirrors.aliyun.com/debian/ bookworm-updates main" >> /etc/apt/sources.list \
    && echo "deb https://mirrors.aliyun.com/debian-security bookworm-security main" >> /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    -o Acquire::http::Timeout="120" \
    -o Acquire::http::Max-Retries="5" \
    tzdata openntpd \
    && ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# 挂载卷 docker run -d -v /home/xmlq/codes/pycodes/nonebot/lunabot-sekai/data:/app/lunabot_event_tracker/data --name lunabot_event_tracker lunabot_event_tracker:latest
VOLUME ["/app/lunabot_event_tracker/config", "/app/lunabot_event_tracker/data"]

# 启动服务
CMD ["python", "main.py"]