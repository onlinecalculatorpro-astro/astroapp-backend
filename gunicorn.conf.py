# gunicorn.conf.py
import multiprocessing, os

bind = f"0.0.0.0:{os.getenv('PORT','5000')}"
workers = max(2, multiprocessing.cpu_count())  # CPU-bound astro math: prefer more workers
threads = 1
worker_class = "sync"
timeout = 90
graceful_timeout = 30
keepalive = 2
accesslog = "-"   # stdout
errorlog = "-"    # stderr
loglevel = os.getenv("LOGLEVEL", "info")

# add request id if present
access_log_format = (
    '%(h)s - "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" '
    'req_id:%({X-Request-ID}i)s rt:%(L)s'
)
