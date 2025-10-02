ENABLE_PROXY_FIX = False

# CSRF/сессии
WTF_CSRF_ENABLED = False
WTF_CSRF_SSL_STRICT = False
WTF_CSRF_TIME_LIMIT = None

# Локальная разработка по HTTP:
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_SECURE = False
SESSION_COOKIE_DOMAIN = None  # привязывает куку к конкретному host (IP)


# УБРАТЬ CORS
ENABLE_CORS = False
# (и можно удалить CORS_OPTIONS, если был)
