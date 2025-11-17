package assets

import thirdservice "lunabot/xmlq/server/third_service"

type ApiGroup struct {
	MasterdataApi
	AssetApi
	MusicApi
}

// 适配的数据源
const (
	HARUKI = "haruki"
)

var (
	harukiMasterdataService = thirdservice.ThirdServiceApp.HarukiApiGroup.MasterdataService
	harukiAssetService      = thirdservice.ThirdServiceApp.HarukiApiGroup.AssetService
	harukiMusicService      = thirdservice.ThirdServiceApp.HarukiApiGroup.MusicService
)
