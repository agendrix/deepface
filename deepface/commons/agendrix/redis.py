from os import environ

import redis


def initialize_redis():
    return redis.Redis(
        host=environ.get("REDIS_HOST", "localhost"),
        port=int(environ.get("REDIS_PORT", 0)),
        username=environ.get("REDIS_USERNAME", None),
        password=environ.get("REDIS_PASSWORD", None),
        ssl=True,
        ssl_certfile=environ.get("REDIS_SSL_CERTFILE", None),
        ssl_keyfile=environ.get("REDIS_SSL_KEYFILE", None),
        ssl_ca_certs=environ.get("REDIS_SSL_CA_CERTS", None),
    )
