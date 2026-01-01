package system

import "lunabot/xmlq/server/service"

type ApiGroup struct {
	SystemApi
	OperationRecordApi
}

var (
	systemConfigService    = service.ServiceGroupApp.SystemServiceGroup.SystemConfigService
	operationRecordService = service.ServiceGroupApp.SystemServiceGroup.OperationRecordService
)
