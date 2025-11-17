package initialize

import (
	"net/http"
	"os"

	"lunabot/xmlq/server/global"
	"lunabot/xmlq/server/router"

	"github.com/gin-gonic/gin"
)

type justFilesFilesystem struct {
	fs http.FileSystem
}

func (fs justFilesFilesystem) Open(name string) (http.File, error) {
	f, err := fs.fs.Open(name)
	if err != nil {
		return nil, err
	}

	stat, err := f.Stat()
	if stat.IsDir() {
		return nil, os.ErrPermission
	}

	return f, nil
}

// 初始化总路由

func Routers() *gin.Engine {
	Router := gin.New()
	Router.Use(gin.Recovery())
	if gin.Mode() == gin.DebugMode {
		Router.Use(gin.Logger())
	}

	Router.StaticFS(global.CONFIG.Upload.StorePath, justFilesFilesystem{http.Dir(global.CONFIG.Upload.StorePath)}) // Router.Use(middleware.LoadTls())  // 如果需要使用https 请打开此中间件 然后前往 core/server.go 将启动模式 更变为 Router.RunTLS("端口","你的cre/pem文件","你的key文件")
	// 跨域，如需跨域可以打开下面的注释
	// Router.Use(middleware.Cors()) // 直接放行全部跨域请求
	// Router.Use(middleware.CorsByRules()) // 按照配置的规则放行跨域请求
	// global.LOG.Info("use middleware cors")

	PrefixRouter := Router.Group(global.CONFIG.System.RouterPrefix)
	{
		// 健康监测
		PrefixRouter.GET("/health", func(c *gin.Context) {
			c.JSON(http.StatusOK, "ok")
		})
	}
	{
		gameRouter := router.RouterGroupApp.Game
		gameRouter.InitUserRoot(PrefixRouter)
		gameRouter.InitEventRouter(PrefixRouter)

	}
	{
		assetsRouter := router.RouterGroupApp.Assets
		assetsRouter.InitMasterdataRouter(PrefixRouter)
		assetsRouter.InitAssetRouter(PrefixRouter)
		assetsRouter.InitMusicRouter(PrefixRouter)
	}
	global.ROUTERS = Router.Routes()
	global.LOG.Info("router register success")
	return Router
}
