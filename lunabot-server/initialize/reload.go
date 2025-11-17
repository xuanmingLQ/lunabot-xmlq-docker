package initialize

import (
	"lunabot/xmlq/server/global"

	"go.uber.org/zap"
)

// Reload 优雅地重新加载系统配置
func Reload() error {
	global.LOG.Info("正在重新加载系统配置...")

	// 重新加载配置文件
	if err := global.VP.ReadInConfig(); err != nil {
		global.LOG.Error("重新读取配置文件失败!", zap.Error(err))
		return err
	}

	// 重新初始化其他配置
	OtherInit()

	global.LOG.Info("系统配置重新加载完成")
	return nil
}
