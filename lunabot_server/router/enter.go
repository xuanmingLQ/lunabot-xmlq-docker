package router

import (
	"lunabot/xmlq/server/router/assets"
	"lunabot/xmlq/server/router/game"
	"lunabot/xmlq/server/router/system"
)

var RouterGroupApp = new(RouterGroup)

type RouterGroup struct {
	System system.RouterGroup
	Game   game.RouterGroup
	Assets assets.RouterGroup
}
