package game

import api "lunabot/xmlq/server/api/v1"

type RouterGroup struct {
	UserRouter
	EventRouter
}

var (
	userApi  = api.ApiGroupApp.GameApiGroup.UserApi
	eventApi = api.ApiGroupApp.GameApiGroup.EventApi
)
