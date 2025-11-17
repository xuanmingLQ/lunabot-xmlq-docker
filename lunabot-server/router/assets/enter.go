package assets

import api "lunabot/xmlq/server/api/v1"

type RouterGroup struct {
	MasterdataRouter
	AssetRouter
	MusicRouter
}

var (
	masterdataApi = api.ApiGroupApp.AssetsApiGroup.MasterdataApi
	assetApi        = api.ApiGroupApp.AssetsApiGroup.AssetApi
	musicApi      = api.ApiGroupApp.AssetsApiGroup.MusicApi
)
