package thirdservice

import "lunabot/xmlq/server/third_service/harukiapi"

type ThirdServiceGroup struct {
	HarukiApiGroup harukiapi.ServiceGroup
}

var ThirdServiceApp = ThirdServiceGroup{}
