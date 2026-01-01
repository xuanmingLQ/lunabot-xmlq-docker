package global

import (
	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
	"gorm.io/gorm"

	"go.uber.org/zap"

	"lunabot/xmlq/server/config"
	"lunabot/xmlq/server/utils/timer"

	"github.com/spf13/viper"
)

var (
	CONFIG config.Server
	VP     *viper.Viper
	// LOG    *oplogging.Logger
	LOG     *zap.Logger
	ROUTERS gin.RoutesInfo
	DBNAME  *string
	DB      *gorm.DB              // 持久化数据库
	REDIS   redis.UniversalClient // 内存数据库
	Timer   timer.Timer           = timer.NewTimerTask()
)
