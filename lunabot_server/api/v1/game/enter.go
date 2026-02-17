package game

import (
	"fmt"
	"lunabot/xmlq/server/service"
	thirdservice "lunabot/xmlq/server/third_service"
	"lunabot/xmlq/server/third_service/harukiapi"
	"lunabot/xmlq/server/utils"
)

type ApiGroup struct {
	UserApi
	EventApi
}

var (
	harukiApiService = thirdservice.ThirdServiceApp.HarukiApiGroup.GameApiService
	suiteService     = service.ServiceGroupApp.GameServiceGroup.SuiteService
	mysekaiService   = service.ServiceGroupApp.GameServiceGroup.MysekaiService
)

// 不在返回的消息中携带url
func resultError(err error) string {
	if hae, ok := err.(*harukiapi.HarukiApiError); ok {
		if hae.Status == 404 {
			return "未绑定 Haruki 工具箱或未验证"
		}
		if hae.Status == 403 {
			return "没有在 Haruki 工具箱允许公开 Api"
		}
		return fmt.Sprintf("%d: %s", hae.Status, hae.Message)
	}
	if he, ok := err.(*utils.HttpError); ok {
		return fmt.Sprintf("%d: %s", he.StatusCode, he.Detail)
	}
	return err.Error()
}
