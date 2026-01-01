package jp

import (
	base "lunabot/xmlq/server/model/game/base"
)

type Suite struct {
	base.Suite
}

func (Suite) TableName() string {
	return "jp_suite"
}
