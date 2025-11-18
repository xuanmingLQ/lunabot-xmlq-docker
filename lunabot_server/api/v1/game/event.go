package game

import (
	"lunabot/xmlq/server/global"
	"lunabot/xmlq/server/model/common/response"
	"lunabot/xmlq/server/model/game/request"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

type EventApi struct{}

// GetRanking
// @Summary 获取ranking数据
// @Produce application/json
// @Param data query game.Event
// @Success 200 {object}
// @Router /event/ranking [get]
func (*EventApi) GetRanking(c *gin.Context) {
	var eventInfo request.Event
	if err := c.ShouldBindQuery(&eventInfo); err != nil {
		global.LOG.Error("参数校验不通过！", zap.Error(err))
		response.FailWithMessage("参数校验不通过", c)
		return
	}
	if ranking, err := harukiApiService.GetRanking(c, eventInfo.Region, eventInfo.EventId); err != nil {
		global.LOG.Error("请求 Haruki Api 获取 Ranking 数据失败！", zap.Error(err))
		response.FailWithMessage("请求 Haruki Api 获取 Ranking 数据失败！", c)
		return
	} else {
		response.OkWithData(ranking, c)
		return
	}
}
