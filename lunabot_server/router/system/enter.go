package system

import api "lunabot/xmlq/server/api/v1"

type RouterGroup struct {
	OperationRecordRouter
	SysRouter
}

var (
	systemApi          = api.ApiGroupApp.SystemApiGroup.SystemApi
	operationRecordApi = api.ApiGroupApp.SystemApiGroup.OperationRecordApi
)
