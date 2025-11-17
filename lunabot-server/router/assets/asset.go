package assets

import "github.com/gin-gonic/gin"

type AssetRouter struct{}

func (*AssetRouter) InitAssetRouter(Router *gin.RouterGroup) {
	assetRouter := Router.Group("asset")
	{
		assetRouter.GET("downloadAsset/:region/*path", assetApi.DownloadAsset)
	}
}
