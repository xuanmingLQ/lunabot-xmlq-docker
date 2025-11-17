package core

import (
	"fmt"
	"lunabot/xmlq/server/global"
	"lunabot/xmlq/server/initialize"
	"time"
)

func RunServer() {

	Router := initialize.Routers()

	address := fmt.Sprintf(":%d", global.CONFIG.System.Addr)

	fmt.Printf(`lunabot xmlq server running
当前版本: %s
地址: http://127.0.0.1%s
`, global.Version, address)
	initServer(address, Router, 10*time.Minute, 10*time.Minute)
}
