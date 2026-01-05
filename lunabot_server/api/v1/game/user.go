package game

import (
	"context"
	"encoding/json"
	"fmt"
	"lunabot/xmlq/server/global"
	"lunabot/xmlq/server/model/common/response"
	"lunabot/xmlq/server/model/game/base"
	"lunabot/xmlq/server/model/game/request"
	gameRes "lunabot/xmlq/server/model/game/response"
	"lunabot/xmlq/server/third_service/harukiapi"
	"maps"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

type UserApi struct{}

// GetProfile
// @Summary 获取profile数据
// @Produce application/json
// @Param data query game.User
// @Success 200 {object}
// @Router /user/profile [get]
func (*UserApi) GetProfile(c *gin.Context) {
	var userInfo request.User
	if err := userInfo.BindQuery(c); err != nil {
		global.LOG.Error("参数校验失败！", zap.Error(err))
		response.FailWithMessage(fmt.Sprintf("参数校验失败 %s", err.Error()), c)
		return
	}
	if profile, err := harukiApiService.GetProfile(c, userInfo.Region, userInfo.UserId); err != nil {
		global.LOG.Error(fmt.Sprintf("请求 Haruki Api 获取 %s 的 Profile 数据失败！", userInfo.UserId), zap.Error(err))
		response.FailWithMessage("请求 Haruki Api 获取 Profile 数据失败！", c)
		return
	} else {
		response.OkWithData(profile, c)
		return
	}
}

// GetSuite
// @Summary 获取Suite数据
// @Produce application/json
// @Param data query game.User
// @Success 200 {object}
// @Router /user/suite [get]
func (*UserApi) GetSuite(c *gin.Context) {
	var userInfo request.User
	if err := userInfo.BindQuery(c); err != nil {
		global.LOG.Error("参数校验失败！", zap.Error(err))
		response.FailWithMessage(fmt.Sprintf("参数校验失败 %s", err.Error()), c)
		return
	}
	filters := c.QueryArray("filters")
	filters = append(filters, "")
	// 首先检查本地的suite和haruki的suite上传时间GetSuite
	idTime, localErr := suiteService.GetUploadTime(c, userInfo.Region, userInfo.UserId)
	localUploadTime := idTime[userInfo.UserId]
	if localErr != nil {
		global.LOG.Error(fmt.Sprintf("获取 %s 本地的 suite 上传时间失败", userInfo.UserId), zap.Error(localErr))
	}
	harukiUploadTime, harukiErr := harukiApiService.GetSuiteUploadTime(c, userInfo.Region, userInfo.UserId)
	if harukiErr != nil {
		global.LOG.Error(fmt.Sprintf("从 Haruki 工具箱获取 %s 的 suite 上传时间失败", userInfo.UserId), zap.Error(harukiErr))
		// 如果本地和Haruki都失败
		if localErr != nil {
			response.FailWithMessage("获取 Suite 数据失败："+resultError(harukiErr), c)
			return
		}
	}

	// 如果本地的上传时间较新，或者haruki没有拿到上传时间
	if localUploadTime != nil && *localUploadTime != 0 && (harukiUploadTime == nil || *localUploadTime > *harukiUploadTime) {
		suite, err := suiteService.GetDataWithFilter(c, userInfo)
		if err != nil {
			// 本地获取失败可以再尝试从haruki获取一次，因为可能本地数据有问题
			global.LOG.Error(fmt.Sprintf("获取 %s 本地的 suite 数据失败", userInfo.UserId), zap.Error(err))
		} else {
			for _, f := range userInfo.Filter {
				if v := suite[f]; v == nil {
					err = fmt.Errorf("本地的 suite 数据中缺少 %s", f)
					break
				}
			}
			if err != nil {
				global.LOG.Error(err.Error())
			} else {
				response.OkWithData(suite, c)
				return
			}
		}
	}
	// 如果haruki上传时间较新，从Haruki获取所有数据之后，保存到本地，同时把请求的数据返回
	suite, err := harukiApiService.GetSuite(c, userInfo.Region, userInfo.UserId)
	if err != nil {
		global.LOG.Error(fmt.Sprintf("从 Haruki 工具箱获取 %s 的 suite 数据失败", userInfo.UserId), zap.Error(err))
		response.FailWithMessage(fmt.Sprintf("从 Haruki 工具箱获取 suite 数据失败 %s", err.Error()), c)
		return
	}
	source, ok := suite["source"].(string)
	if !ok {
		source = harukiapi.HARUKI
	}
	// 返回所需的数据
	resultSuite := make(map[string]interface{})
	filter := userInfo.Filter
	if len(filter) == 0 {
		filter = global.CONFIG.HarukiApi.SuiteApi.DefaultSuiteKeys
	}
	for _, f := range filter {
		resultSuite[f] = suite[f]
	}
	resultSuite["source"] = source
	response.OkWithData(resultSuite, c)

	// 保存suite到本地
	suite["source"] = fmt.Sprintf("Local(%s)", source)
	err = suiteService.Save(c, userInfo.Region, base.Suite{
		UserId: json.Number(userInfo.UserId),
		Data:   suite,
	})
	if err != nil {
		global.LOG.Error("将 suite 数据保存到本地失败", zap.Error(err))
	}
}

// GetSuiteUploadTime
// @Summary 查询单个用户的suite数据上传时间
// @Produce application/json
// @Param data query {region string, userId string}
// @Success 200 {object}
// @Router /user/suiteUploadTime [get]
func (*UserApi) GetSuiteUploadTime(c *gin.Context) {
	var userInfo request.User
	if err := userInfo.BindQuery(c); err != nil {
		global.LOG.Error("参数校验失败！", zap.Error(err))
		response.FailWithMessage(fmt.Sprintf("参数校验失败 %s", err.Error()), c)
		return
	}
	idTime, localErr := suiteService.GetUploadTime(c, userInfo.Region, userInfo.UserId)
	localUploadTime, ok := idTime[userInfo.UserId]
	if !ok {
		localErr = fmt.Errorf("本地没有 %s 的 suite 数据", userInfo.UserId)
	}
	if localErr != nil {
		global.LOG.Error("获取本地的 suite 上传时间失败", zap.Error(localErr))
	}
	harukiUploadTime, harukiErr := harukiApiService.GetSuiteUploadTime(c, userInfo.Region, userInfo.UserId)
	if harukiErr != nil {
		global.LOG.Error("从 Haruki 工具箱获取 suite 上传时间失败", zap.Error(harukiErr))
	}

	response.OkWithData(
		gin.H{
			"本地数据":       gameRes.UploadTime{UploadTime: localUploadTime, Error: localErr},
			"Haruki 工具箱": gameRes.UploadTime{UploadTime: harukiUploadTime, Error: harukiErr},
		}, c)
	// 如果haruki的数据较新，就去get一下
	if harukiUploadTime != nil && *harukiUploadTime != 0 && (localUploadTime == nil || *harukiUploadTime > *localUploadTime) {

		suite, err := harukiApiService.GetSuite(c, userInfo.Region, userInfo.UserId)
		if err != nil {
			global.LOG.Error("从 Haruki 工具箱获取 suite 数据失败", zap.Error(err))
			return
		}
		source, ok := suite["source"].(string)
		if !ok {
			source = harukiapi.HARUKI
		}
		suite["source"] = fmt.Sprintf("Local(%s)", source)
		err = suiteService.Save(c, userInfo.Region, base.Suite{
			UserId: json.Number(userInfo.UserId),
			Data:   suite,
		})
		if err != nil {
			global.LOG.Error("将 suite 数据保存到本地失败", zap.Error(err))
		}
	}
}

// 从haruki api获取烤森数据，保存到本地，然后返回所需数据
func getAndSaveMysekai(ctx context.Context, RequestUser request.User) (result map[string]interface{}, err error) {
	mysekai, err := harukiApiService.GetMysekai(ctx, RequestUser.Region, RequestUser.UserId)
	if err != nil {
		return
	}
	source, ok := mysekai["source"].(string)
	if !ok {
		source = harukiapi.HARUKI
	}
	// 另开一个goroutine用来保存到本地
	go func() {
		// 保存到本地
		mysekai["source"] = fmt.Sprintf("Local(%s)", source)
		err = mysekaiService.Save(ctx, RequestUser.Region, base.Mysekai{
			UserId: json.Number(RequestUser.UserId),
			Data:   mysekai,
		})
		if err != nil {
			global.LOG.Error("将 mysekai 数据保存到本地失败", zap.Error(err))
		}
	}()
	// 返回所需数据
	result = make(map[string]interface{}, len(mysekai))
	if len(RequestUser.Filter) == 0 {
		maps.Copy(result, mysekai)
	} else {
		for _, f := range RequestUser.Filter {
			result[f] = mysekai[f]
		}
	}
	result["source"] = source
	return result, nil
}

// GetMysekai
// @Summary 获取Mysekai数据
// @Produce application/json
// @Param data query game.User
// @Success 200 {object}
// @Router /user/mysekai [get]
func (*UserApi) GetMysekai(c *gin.Context) {
	var userInfo request.User
	if err := userInfo.BindQuery(c); err != nil {
		global.LOG.Error("参数校验失败！", zap.Error(err))
		response.FailWithMessage(fmt.Sprintf("参数校验失败 %s", err.Error()), c)
		return
	}
	// 检查本地的和haruki的上传时间
	idTime, localErr := mysekaiService.GetUploadTime(c, userInfo.Region, userInfo.UserId)
	localUploadTime := idTime[userInfo.UserId]
	if localErr != nil {
		global.LOG.Error(fmt.Sprintf("获取 %s 本地的 mysekai 上传时间失败", userInfo.UserId), zap.Error(localErr))
	}
	harukiUploadTime, harukiErr := harukiApiService.GetMysekaiUploadTime(c, userInfo.Region, userInfo.UserId)
	if harukiErr != nil {
		global.LOG.Error(fmt.Sprintf("从 Haruki 工具箱获取 %s 的 mysekai 上传时间失败", userInfo.UserId), zap.Error(harukiErr))
		// 如果本地和Haruki都失败
		if localErr != nil {
			response.FailWithMessage("获取 Suite 数据失败："+resultError(harukiErr), c)
			return
		}
	}
	// 本地上传时间较新
	if localUploadTime != nil && *localUploadTime != 0 && (harukiUploadTime == nil || *localUploadTime > *harukiUploadTime) {
		mysekai, err := mysekaiService.GetDataWithFilter(c, userInfo)
		if err != nil {
			global.LOG.Error(fmt.Sprintf("获取 %s 本地的 mysekai 数据失败", userInfo.UserId), zap.Error(err))
		} else {
			response.OkWithData(mysekai, c)
		}
		return
	}
	result, err := getAndSaveMysekai(c, userInfo)
	if err != nil {
		global.LOG.Error(fmt.Sprintf("从 Haruki 工具箱获取 %s 的 mysekai 数据失败", userInfo.UserId), zap.Error(err))
		response.FailWithMessage(fmt.Sprintf("从 Haruki 工具箱获取 mysekai 数据失败: %s", err.Error()), c)
		return
	}
	response.OkWithData(result, c)
}

// GetMysekaiUploadTime
// @Summary 查询单个用户的Mysekai数据上传时间
// @Produce application/json
// @Param data query {region string, userId string}
// @Success 200 {object}
// @Router /user/mysekaiUploadTime [get]
func (*UserApi) GetMysekaiUploadTime(c *gin.Context) {
	var userInfo request.User
	if err := userInfo.BindQuery(c); err != nil {
		global.LOG.Error("参数校验失败！", zap.Error(err))
		response.FailWithMessage(fmt.Sprintf("参数校验失败 %s", err.Error()), c)
		return
	}
	idTime, localErr := mysekaiService.GetUploadTime(c, userInfo.Region, userInfo.UserId)
	localUploadTime, ok := idTime[userInfo.UserId]
	if !ok {
		localErr = fmt.Errorf("本地没有 %s 的 suite 数据", userInfo.UserId)
	}
	if localErr != nil {
		global.LOG.Error(fmt.Sprintf("获取 %s 本地的 mysekai 上传时间失败", userInfo.UserId), zap.Error(localErr))
	}
	harukiUploadTime, harukiErr := harukiApiService.GetMysekaiUploadTime(c, userInfo.Region, userInfo.UserId)
	if harukiErr != nil {
		global.LOG.Error(fmt.Sprintf("从 Haruki 工具箱获取 %s 的 mysekai 上传时间失败", userInfo.UserId), zap.Error(harukiErr))
	}
	response.OkWithData(
		gin.H{
			"本地数据":       gameRes.UploadTime{UploadTime: localUploadTime, Error: localErr},
			"Haruki 工具箱": gameRes.UploadTime{UploadTime: harukiUploadTime, Error: harukiErr},
		}, c)
	// 如果haruki的数据较新，将它保存在本地
	if harukiUploadTime != nil && *harukiUploadTime != 0 && (localUploadTime == nil || *harukiUploadTime > *localUploadTime) {
		_, err := getAndSaveMysekai(c, userInfo)
		if err != nil {
			global.LOG.Error(fmt.Sprintf("从 Haruki 工具箱获取 %s 的 mysekai 数据失败", userInfo.UserId), zap.Error(err))
			return
		}
	}
}

// MysekaiUploadTime
// @Summary 查询多个用户的Mysekai数据上传时间
// @Produce application/json
// @Param data json gameReq.Users
// @Success 200 {object}
// @Router /user/mysekaiUploadTime [put]
func (*UserApi) MysekaiUploadTime(c *gin.Context) {
	var users request.Users
	err := c.ShouldBindJSON(&users)
	if err != nil {
		response.FailWithMessage(fmt.Sprintf("参数校验失败：%s", err.Error()), c)
		return
	}
	userIds := make([]string, len(users.UserIds))
	for i, userId := range users.UserIds {
		userIdStr := userId.String()
		userIds[i] = userIdStr
	}
	harukiUploadTimes, harukiErr := harukiApiService.GetMysekaiUploadTimeByIds(c, users.Region, userIds...)
	if harukiErr != nil {
		// 由于harukiapi失败了，无法比较
		global.LOG.Error("从 Haruki 工具箱获取 mysekai 上传时间失败", zap.Error(harukiErr))
		response.FailWithMessage(
			fmt.Sprintf("从 Haruki 工具箱获取 mysekai 上传时间失败: %s", harukiErr.Error()),
			c)
		return
	}
	// 检查本地的上传时间
	localUploadTimes, localErr := mysekaiService.GetUploadTime(c, users.Region, userIds...)
	if localErr != nil {
		global.LOG.Error("获取本地的 mysekai 上传时间失败", zap.Error(localErr))
	}
	result := make(map[json.Number]*int64, len(userIds))
	for _, userId := range userIds {
		userIdNumber := json.Number(userId)
		harukiUploadTime := harukiUploadTimes[userId]
		localUploadTime := localUploadTimes[userId]
		// 保存其中较新的时间，同时如果haruki的时间较新，获取并保存到数据库
		if harukiUploadTime != nil && *harukiUploadTime != 0 && (localUploadTime == nil || *harukiUploadTime > *localUploadTime) {
			result[userIdNumber] = harukiUploadTime
			// 开启一个goroutine去保存数据
			go getAndSaveMysekai(c, request.User{
				Region: users.Region,
				UserId: userId,
			})
			continue
		}
		if localUploadTime != nil {
			result[userIdNumber] = localUploadTime
		}
	}
	if len(result) == 0 {
		global.LOG.Error("获取 mysekai 上传时间全部失败")
	}
	// 将比较后的时间返回
	response.OkWithData(result, c)
}
