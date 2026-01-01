package initialize

import (
	"context"
	"lunabot/xmlq/server/service/system"

	"lunabot/xmlq/server/model/game/cn"
	"lunabot/xmlq/server/model/game/jp"
	sysModel "lunabot/xmlq/server/model/system"

	"gorm.io/gorm"
)

const initOrderEnsureTables = system.InitOrderExternal - 1

type ensureTables struct{}

// auto run
func init() {
	system.RegisterInit(initOrderEnsureTables, &ensureTables{})
}

func (e *ensureTables) InitializerName() string {
	return "ensure_tables_created"
}
func (e *ensureTables) InitializeData(ctx context.Context) (next context.Context, err error) {
	return ctx, nil
}

func (e *ensureTables) DataInserted(ctx context.Context) bool {
	return true
}

// 自动迁移表，每当新增一个表都给它放进来
func (e *ensureTables) MigrateTable(ctx context.Context) (context.Context, error) {
	db, ok := ctx.Value("db").(*gorm.DB)
	if !ok {
		return ctx, system.ErrMissingDBContext
	}
	tables := []interface{}{
		sysModel.SysOperationRecord{},
		cn.Suite{},
		cn.Mysekai{},
		jp.Suite{},
		jp.Mysekai{},
	}
	for _, t := range tables {
		_ = db.AutoMigrate(&t)
		// 由于 AutoMigrate() 基本无需考虑错误，因此显式忽略
	}
	return ctx, nil
}

// 自动创建表
func (e *ensureTables) TableCreated(ctx context.Context) bool {
	db, ok := ctx.Value("db").(*gorm.DB)
	if !ok {
		return false
	}
	tables := []interface{}{
		sysModel.SysOperationRecord{},
		cn.Suite{},
		cn.Mysekai{},
		jp.Suite{},
		jp.Mysekai{},
	}
	yes := true
	for _, t := range tables {
		yes = yes && db.Migrator().HasTable(t)
	}
	return yes
}
