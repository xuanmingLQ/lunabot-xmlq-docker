package request

import (
	"lunabot/xmlq/server/model/common/request"
	"lunabot/xmlq/server/model/system"
)

type SysOperationRecordSearch struct {
	system.SysOperationRecord
	request.PageInfo
}
