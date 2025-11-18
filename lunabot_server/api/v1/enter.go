package v1

import (
	"lunabot/xmlq/server/api/v1/assets"
	"lunabot/xmlq/server/api/v1/game"
)

var ApiGroupApp = new(ApiGroup)

type ApiGroup struct {
	GameApiGroup   game.ApiGroup
	AssetsApiGroup assets.ApiGroup
}
