FROM python:3.12-slim

# XGBoost 在 Linux 上需要 OpenMP 运行库
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖，再复制代码：代码改了也不用重新装包，构建更快
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ api/
COPY artifacts/ artifacts/

EXPOSE 8000
# 部署平台（如 Render）会通过 PORT 环境变量指定端口；本地默认 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]