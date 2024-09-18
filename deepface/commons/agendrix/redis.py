from os import environ

import redis


def initialize_redis():
    return redis.Redis(
        host=environ.get("REDIS_HOST", "localhost"),
        port=int(environ.get("REDIS_PORT", 0)),
    )
