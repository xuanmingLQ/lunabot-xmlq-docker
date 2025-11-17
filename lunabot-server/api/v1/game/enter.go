package game

import thirdservice "lunabot/xmlq/server/third_service"

type ApiGroup struct {
	UserApi
	EventApi
}

var (
	harukiApiService   = thirdservice.ThirdServiceApp.HarukiApiGroup.GameApiService
)
