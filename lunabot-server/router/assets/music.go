package assets

import "github.com/gin-gonic/gin"

type MusicRouter struct{}

func (*MusicRouter) InitMusicRouter(Router *gin.RouterGroup) {
	musicRouter := Router.Group("music")
	{
		musicRouter.GET("alias", musicApi.GetAlias)
	}
}
