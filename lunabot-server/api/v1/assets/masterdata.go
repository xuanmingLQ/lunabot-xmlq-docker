package assets

import (
	"context"
	"fmt"
	"lunabot/xmlq/server/global"
	"lunabot/xmlq/server/model/assets/request"
	"lunabot/xmlq/server/model/common/response"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

type MasterdataApi struct{}

// UpdateVersion
// @Summary 获取当前的版本
// @Produce application/json
// @Param data query []string
// @Success 200 {object}
// @Router /masterdata/version [get]
func (*MasterdataApi) GetVersion(c *gin.Context) {
	region, _ := c.GetQueryArray("region")
	var Regions []string
	if len(region) == 0 || (len(region) == 1 && region[0] == "all") {
		// 更新所有服务器的版本
		for Region := range global.CONFIG.SekaiAsset.Sources {
			Regions = append(Regions, Region)
		}
	} else {
		Regions = region
	}
	// 返回值
	var result = make(map[string]interface{})
	for _, region := range Regions { //对于请求的每个服务器
		sources, ok := global.CONFIG.SekaiAsset.Sources[region]
		if !ok { //如果没有配置该服务器就跳过
			continue
		}
		var resultRegion = make(map[string]interface{})
		for sourceName, masterdataSource := range sources.Masterdata {
			if masterdataSource.VersionUrl == "" {
				continue
			}
			var versionData interface{}
			var err error
			switch sourceName {
			case HARUKI:
				versionData, err = harukiMasterdataService.GetCurrentVersion(c, masterdataSource.VersionUrl)
			default:
				global.LOG.Warn(fmt.Sprintf("数据源: %s 没有指定获取方法", sourceName))
				continue
			}
			if err != nil {
				global.LOG.Error(fmt.Sprintf("从数据源：%s 获取Masterdata Version失败", sourceName), zap.Error(err))
			} else {
				resultRegion[sourceName] = versionData
			}
		}
		if len(resultRegion) > 0 {
			result[region] = resultRegion
		}
	}
	if len(region) == 1 && region[0] != "all" { //如果只请求了一个数据，那就直接把这一个数据放出来
		if resultRegion, ok := result[region[0]]; ok {
			result = resultRegion.(map[string]interface{})
		}
	}
	if len(result) > 0 {
		response.OkWithData(result, c)
	} else {
		response.FailWithMessage("从所有数据源获取Masterdata Version失败", c)
	}
}

// DownloadMasterdata
// @Summary 下载Masterdatta
// @Produce application/json
// @Param data query request.Masterdata
// @Success 200 {object}
// @Router /masterdata/download [get]
func (*MasterdataApi) DownloadMasterdata(c *gin.Context) {
	var requestMasterdata request.Masterdata
	if err := c.ShouldBindQuery(&requestMasterdata); err != nil {
		global.LOG.Error("参数校验不通过！", zap.Error(err))
		response.FailWithMessage("参数校验不通过", c)
		return
	}
	var downloadFunc func(ctx context.Context, BaseUrl, Name string) (interface{}, error)
	switch requestMasterdata.Source {
	case HARUKI:
		downloadFunc = harukiMasterdataService.DownloadMasterdata
	default:
		global.LOG.Warn(fmt.Sprintf("数据源: %s 没有指定方法", requestMasterdata.Source))
		response.FailWithMessage(fmt.Sprintf("数据源: %s 没有指定方法", requestMasterdata.Source), c)
		return
	}
	source := global.CONFIG.SekaiAsset.Sources[requestMasterdata.Region].Masterdata[requestMasterdata.Source]
	if source.BaseUrl == "" {
		global.LOG.Warn(fmt.Sprintf("数据源：%s 没有配置Masterdata BaseUrl", requestMasterdata.Source))
		response.FailWithMessage(fmt.Sprintf("数据源：%s 没有配置Masterdata BaseUrl", requestMasterdata.Source), c)
		return
	}
	var result interface{}
	resultMap := make(map[string]interface{})
	for _, name := range requestMasterdata.Name {
		var masterdata interface{}
		var err error
		masterdata, err = downloadFunc(c, source.BaseUrl, name)
		if err != nil {
			global.LOG.Error(fmt.Sprintf("从数据源：%s 获取Masterdata: %s 失败", requestMasterdata.Source, name), zap.Error(err))
			continue
		} else {
			resultMap[name] = masterdata
		}

	}
	if len(requestMasterdata.Name) == 1 { // 如果只请求了一个数据，那就直接把这一个数据放出来
		result = resultMap[requestMasterdata.Name[0]]
	} else if len(resultMap) > 0 { // 如果请求的不止一个数据，且确实请求到了数据，就返回完整数据
		result = resultMap
	}
	if result != nil { // 如果确实请求到了数据，返回这些数据
		response.OkWithData(result, c)
	} else {
		response.FailWithMessage("从所有数据源获取Masterdata失败", c)
	}
}
