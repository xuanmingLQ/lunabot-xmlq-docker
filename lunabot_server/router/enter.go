package router

import (
	"lunabot/xmlq/server/router/assets"
	"lunabot/xmlq/server/router/game"
)

var RouterGroupApp = new(RouterGroup)

type RouterGroup struct {
	Game   game.RouterGroup
	Assets assets.RouterGroup
}
