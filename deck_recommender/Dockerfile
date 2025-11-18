FROM python:3.13-slim-bookworm AS builder

# 设置工作目录
WORKDIR /app/deck_recommender

# 将 requirements.txt 复制到工作目录
COPY requirements.txt .

# 安装 Python 依赖
# 使用 --no-cache-dir 减少缓存大小
RUN pip install -i https://mirrors.aliyun.com/pypi/simple --no-cache-dir -r requirements.txt

FROM python:3.13-slim-bookworm AS runtime

# 设置工作目录
WORKDIR /app/deck_recommender

# 从构建阶段复制已安装的依赖
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages

# 复制应用程序代码
COPY . .

# 暴露应用程序运行的端口
EXPOSE 45556

# 挂载卷
VOLUME ["/app/deck_recommender/data/sekai/deckrec", "/app/deck_recommender/config.yaml"]

# 启动服务
CMD ["python", "serve.py"]