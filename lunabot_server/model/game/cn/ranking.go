package cn

import (
	base "lunabot/xmlq/server/model/game/base"
)

type Ranking struct {
	base.Ranking
}

func (Ranking) TableName() string {
	return "cn_ranking"
}
