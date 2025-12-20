FROM python:3.13-slim-bookworm

# 1. 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 设置工作目录
WORKDIR /app/lunabot_autochat

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
ENV PATH="/app/lunabot_autochat/.venv/bin:$PATH"

# 挂载卷 docker run -d -v /home/xmlq/codes/pycodes/nonebot/lunabot-sekai/data:/app/deck_recommender/data -p 45556:45556 --name deck-recommender deck-recommender:latest
VOLUME ["/app/lunabot_autochat/config/", "/app/lunabot_autochat/data"]

# 启动服务
CMD ["python", "serve.py"]