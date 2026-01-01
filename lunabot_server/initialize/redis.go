package initialize

import (
	"context"

	"lunabot/xmlq/server/config"
	"lunabot/xmlq/server/global"

	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

func initRedisClient(redisCfg config.Redis) (redis.UniversalClient, error) {
	// 使用单例模式
	client := redis.NewClient(&redis.Options{
		Addr:     redisCfg.Addr,
		Password: redisCfg.Password,
		DB:       redisCfg.DB,
	})

	pong, err := client.Ping(context.Background()).Result()
	if err != nil {
		global.LOG.Error("redis connect ping failed, err:", zap.Error(err))
		return nil, err
	}

	global.LOG.Info("redis connect ping response:", zap.String("pong", pong))
	return client, nil
}

func Redis() {
	redisClient, err := initRedisClient(global.CONFIG.Redis)
	if err != nil {
		panic(err)
	}
	global.REDIS = redisClient
}
