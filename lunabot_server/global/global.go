package global

import (
	"sync"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
	"gorm.io/gorm"

	"go.uber.org/zap"

	"lunabot/xmlq/server/config"

	"github.com/spf13/viper"
)

var (
	CONFIG config.Server
	VP     *viper.Viper
	// LOG    *oplogging.Logger
	LOG       *zap.Logger
	ROUTERS   gin.RoutesInfo
	GVA_DB    *gorm.DB              // 持久化数据库
	GVA_REDIS redis.UniversalClient // 内存数据库
	lock      sync.RWMutex
)
