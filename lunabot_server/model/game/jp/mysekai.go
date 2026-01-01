package jp

import (
	base "lunabot/xmlq/server/model/game/base"
)

type Mysekai struct {
	base.Mysekai
}

func (Mysekai) TableName() string {
	return "jp_mysekai"
}
