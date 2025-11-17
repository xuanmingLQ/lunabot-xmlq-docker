package service

import (
	"lunabot/xmlq/server/service/assets"
	"lunabot/xmlq/server/service/game"
)

var ServiceGroupApp = new(ServiceGroup)

type ServiceGroup struct {
	AssetsServiceGroup assets.ServiceGroup
	GameServiceGroup   game.ServiceGroup
}
