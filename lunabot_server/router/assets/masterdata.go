package assets

import "github.com/gin-gonic/gin"

type MasterdataRouter struct{}

func (*MasterdataRouter) InitMasterdataRouter(Router *gin.RouterGroup) {
	masterdataRouter := Router.Group("masterdata")
	{
		masterdataRouter.GET("version", masterdataApi.GetVersion)
		masterdataRouter.GET("download", masterdataApi.DownloadMasterdata)
	}
}
