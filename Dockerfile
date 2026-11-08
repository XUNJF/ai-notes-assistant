FROM docker.m.daocloud.io/library/python:3.12-slim

WORKDIR /app

# 日志立刻输出，不要攒着（不加的话 docker logs 要等半天）
ENV PYTHONUNBUFFERED=1

# 先只拷依赖清单 → 改代码时这层能用缓存，不用重装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 再拷代码和数据
COPY *.py ./
COPY index.html ./
COPY chroma_db ./chroma_db

# 记忆数据库放这儿（容器里自动建）
RUN mkdir -p data

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]