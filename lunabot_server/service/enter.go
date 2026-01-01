package service

import (
	"lunabot/xmlq/server/service/assets"
	"lunabot/xmlq/server/service/game"
	"lunabot/xmlq/server/service/system"
)

var ServiceGroupApp = new(ServiceGroup)

type ServiceGroup struct {
	SystemServiceGroup system.ServiceGroup
	AssetsServiceGroup assets.ServiceGroup
	GameServiceGroup   game.ServiceGroup
}
