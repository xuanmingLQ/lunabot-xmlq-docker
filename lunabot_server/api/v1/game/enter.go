package game

import (
	"lunabot/xmlq/server/service"
	thirdservice "lunabot/xmlq/server/third_service"
)

type ApiGroup struct {
	UserApi
	EventApi
}

var (
	harukiApiService = thirdservice.ThirdServiceApp.HarukiApiGroup.GameApiService
	suiteService     = service.ServiceGroupApp.GameServiceGroup.SuiteService
	mysekaiService   = service.ServiceGroupApp.GameServiceGroup.MysekaiService
)
