package game

import (
	"maps"
	"context"
	"errors"
	game "lunabot/xmlq/server/model/game"
	gameReq "lunabot/xmlq/server/model/game/request"
)

type EventSerice struct{}

func (*EventSerice) GetRanking(ctx context.Context, Req gameReq.Event) (ranking game.Ranking, err error) {
	a:=map[string]string{}
	var b map[string]string
	maps.Copy(b, a)
	return ranking, errors.New("没有服务")
}
