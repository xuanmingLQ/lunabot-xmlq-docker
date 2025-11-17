package assets

import (
	"lunabot/xmlq/server/global"
	"lunabot/xmlq/server/model/common/response"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

type MusicApi struct{}

// GetAliasByMusicId
// @Summary 获取乐曲昵称
// @Produce application/json
// @Param data query musicId
// @Success 200 {object}
// @Router /music/alias [get]
func (*MusicApi) GetAlias(c *gin.Context) {
	musicId, ok := c.GetQuery("musicId")
	if !ok {
		global.LOG.Error("缺少musicId")
		response.FailWithMessage("缺少musicId", c)
		return
	}
	if result, err := harukiMusicService.GetMusicAlias(c, musicId); err != nil {
		global.LOG.Error("请求 Haruki Api 获取 Music Alias 失败", zap.Error(err))
		response.FailWithMessage("请求 Haruki Api 获取 Music Alias 失败", c)
		return
	} else {
		response.OkWithData(result, c)
		return
	}
}
