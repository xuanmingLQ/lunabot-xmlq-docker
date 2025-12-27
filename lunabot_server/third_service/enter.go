package thirdservice

import (
	"lunabot/xmlq/server/third_service/github"
	"lunabot/xmlq/server/third_service/harukiapi"
	"lunabot/xmlq/server/third_service/unipjsk"
)

type ThirdServiceGroup struct {
	HarukiApiGroup harukiapi.ServiceGroup
	GithubGroup    github.ServiceGroup
	UnipjskGroup   unipjsk.ServiceGroup
}

var ThirdServiceApp = ThirdServiceGroup{}
