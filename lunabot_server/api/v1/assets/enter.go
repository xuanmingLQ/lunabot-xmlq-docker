package assets

import (
	thirdservice "lunabot/xmlq/server/third_service"
)

type ApiGroup struct {
	MasterdataApi
	AssetApi
	MusicApi
}

var (
	githubMasterdataService = thirdservice.ThirdServiceApp.GithubGroup.Masterdata
	harukiAssetsService     = thirdservice.ThirdServiceApp.HarukiApiGroup.AssetsService
	harukiMusicService      = thirdservice.ThirdServiceApp.HarukiApiGroup.MusicService
	unipjskAssetsService    = thirdservice.ThirdServiceApp.UnipjskGroup.AssetsService
)
