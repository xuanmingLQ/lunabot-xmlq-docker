package main

import (
	"lunabot/xmlq/server/core"
	"lunabot/xmlq/server/global"
	"lunabot/xmlq/server/initialize"

	_ "go.uber.org/automaxprocs"
	"go.uber.org/zap"
)

func main() {
	initializeSystem()
	core.RunServer()
}

func initializeSystem() {
	global.VP = core.Viper()
	initialize.OtherInit()
	global.LOG = core.Zap()
	zap.ReplaceGlobals(global.LOG)
	initialize.SetupHandlers()
}
