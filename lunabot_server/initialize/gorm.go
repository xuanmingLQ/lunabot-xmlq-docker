package initialize

import (
	"os"

	"lunabot/xmlq/server/global"
	"lunabot/xmlq/server/model/game/cn"
	"lunabot/xmlq/server/model/game/jp"
	sysModel "lunabot/xmlq/server/model/system"

	"go.uber.org/zap"
	"gorm.io/gorm"
)

func Gorm() *gorm.DB {
	global.DBNAME = &global.CONFIG.Pgsql.Dbname
	return GormPgSql()
}

func RegisterTables() {
	db := global.DB
	err := db.AutoMigrate(
		sysModel.SysOperationRecord{},
		cn.Suite{},
		cn.Mysekai{},
		jp.Suite{},
		jp.Mysekai{},
	)
	if err != nil {
		global.LOG.Error("register table failed", zap.Error(err))
		os.Exit(0)
	}

	if err != nil {
		global.LOG.Error("register biz_table failed", zap.Error(err))
		os.Exit(0)
	}
	global.LOG.Info("register table success")
}
