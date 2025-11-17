package game

import (
	"lunabot/xmlq/server/global"
	"lunabot/xmlq/server/model/common/response"
	"lunabot/xmlq/server/model/game/request"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

type UserApi struct{}

// GetSuite
// @Summary 获取Suite数据
// @Produce application/json
// @Param data query game.User
// @Success 200 {object}
// @Router /user/suite [get]
func (*UserApi) GetSuite(c *gin.Context) {
	var userInfo request.User
	if err := c.ShouldBindQuery(&userInfo); err != nil {
		global.LOG.Error("参数校验不通过！", zap.Error(err))
		response.FailWithMessage("参数校验不通过", c)
		return
	}
	if suiteInfo, err := harukiApiService.GetSuiteInfo(c, userInfo.Region, userInfo.UserId, userInfo.Filter); err != nil {
		global.LOG.Error("请求 Haruki Api 获取 Suite 数据失败！", zap.Error(err))
		response.FailWithMessage("请求 Haruki Api 获取 Suite 数据失败！", c)
		return
	} else {
		response.OkWithData(suiteInfo, c)
		return
	}
}

// GetMysekai
// @Summary 获取Mysekai数据
// @Produce application/json
// @Param data query game.User
// @Success 200 {object}
// @Router /user/mysekai [get]
func (*UserApi) GetMysekai(c *gin.Context) {
	var userInfo request.User
	if err := c.ShouldBindQuery(&userInfo); err != nil {
		global.LOG.Error("参数校验不通过！", zap.Error(err))
		response.FailWithMessage("参数校验不通过", c)
		return
	}
	if mysekaiInfo, err := harukiApiService.GetMysekaiInfo(c, userInfo.Region, userInfo.UserId, userInfo.Filter); err != nil {
		global.LOG.Error("请求 Haruki Api 获取 Mysekai 数据失败！", zap.Error(err))
		response.FailWithMessage("请求 Haruki Api 获取 Mysekai 数据失败！", c)
		return
	} else {
		response.OkWithData(mysekaiInfo, c)
		return
	}
}

// GetProfile
// @Summary 获取profile数据
// @Produce application/json
// @Param data query game.User
// @Success 200 {object}
// @Router /user/profile [get]
func (*UserApi) GetProfile(c *gin.Context) {
	var userInfo request.User
	if err := c.ShouldBindQuery(&userInfo); err != nil {
		global.LOG.Error("参数校验不通过！", zap.Error(err))
		response.FailWithMessage("参数校验不通过", c)
		return
	}
	if profile, err := harukiApiService.GetProfile(c, userInfo.Region, userInfo.UserId); err != nil {
		global.LOG.Error("请求 Haruki Api 获取 Profile 数据失败！", zap.Error(err))
		response.FailWithMessage("请求 Haruki Api 获取 Profile 数据失败！", c)
		return
	} else {
		response.OkWithData(profile, c)
		return
	}
}
