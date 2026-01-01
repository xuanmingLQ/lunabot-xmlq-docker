package game

import (
	"context"
	"errors"
	gameBase "lunabot/xmlq/server/model/game/base"
	gameReq "lunabot/xmlq/server/model/game/request"
	"maps"
)

type EventSerice struct{}

func (*EventSerice) GetRanking(ctx context.Context, Req gameReq.Event) (ranking gameBase.Ranking, err error) {
	a := map[string]string{}
	var b map[string]string
	maps.Copy(b, a)
	return ranking, errors.New("没有服务")
}
