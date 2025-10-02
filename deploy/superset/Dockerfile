# Небольшое расширение официального образа для установки драйвера ClickHouse
FROM apache/superset:latest

# Устанавливаем драйверы и зависимости:
# - clickhouse-connect: современный драйвер + SQLAlchemy диалект
# - psycopg2-binary: клиент для Postgres (метаданные Superset)
RUN pip install --no-cache-dir \
    clickhouse-connect \
    psycopg2-binary
USER root
RUN pip install --no-cache-dir --target /app/pythonpath 'clickhouse-connect>=0.6.8,<1.0'
USER superset
