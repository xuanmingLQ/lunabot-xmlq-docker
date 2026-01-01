package v1

import (
	"lunabot/xmlq/server/api/v1/assets"
	"lunabot/xmlq/server/api/v1/game"
	"lunabot/xmlq/server/api/v1/system"
)

var ApiGroupApp = new(ApiGroup)

type ApiGroup struct {
	SystemApiGroup system.ApiGroup
	GameApiGroup   game.ApiGroup
	AssetsApiGroup assets.ApiGroup
}
