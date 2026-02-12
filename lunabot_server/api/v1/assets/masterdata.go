package assets

import (
	"context"
	"fmt"
	"lunabot/xmlq/server/global"
	"lunabot/xmlq/server/model/assets/request"
	"lunabot/xmlq/server/model/common/response"
	"strings"
	"time"

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
	for _, r := range region {
		if r == "" {
			continue
		}
		for Region := range strings.SplitSeq(r, ",") {
			Region = strings.TrimSpace(Region)
			if Region == "" {
				continue
			}
			Regions = append(Regions, Region)
		}
	}
	// 重新保存展开后的region参数
	region = Regions
	if len(Regions) == 0 || (len(Regions) == 1 && Regions[0] == "all") {
		// 更新所有服务器的版本
		Regions = make([]string, 0, len(global.CONFIG.Masterdata.Sources))
		for Region := range global.CONFIG.Masterdata.Sources {
			Regions = append(Regions, Region)
		}
	}
	// 用context控制超时
	ctx, cacel := context.WithTimeout(c, time.Duration(global.CONFIG.Masterdata.Timeout)*time.Second)
	result, err := githubMasterdataService.GetVersions(ctx, Regions)
	cacel()
	if err != nil {
		global.LOG.Error("获取 Masterdata Version 失败", zap.Error(err))
		response.FailWithMessage("获取 Masterdata Version 失败", c)
		return
	}

	if len(region) == 1 && region[0] != "all" { //如果只请求了一个服务器，那就直接把这一个服务器放出去
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
	if err := requestMasterdata.BindQuery(c); err != nil {
		global.LOG.Error("参数校验不通过！", zap.Error(err))
		response.FailWithMessage("参数校验不通过", c)
		return
	}

	source := global.CONFIG.Masterdata.Sources[requestMasterdata.Region][requestMasterdata.Source]
	if source.BaseUrl == "" {
		global.LOG.Warn(fmt.Sprintf("数据源：%s 没有配置Masterdata BaseUrl", requestMasterdata.Source))
		response.FailWithMessage(fmt.Sprintf("数据源：%s 没有配置Masterdata BaseUrl", requestMasterdata.Source), c)
		return
	}
	var result interface{}
	ctx, cacel := context.WithTimeout(c, time.Duration(global.CONFIG.Masterdata.Timeout)*time.Second)
	masterdatas, err := githubMasterdataService.DownloadMasterdatas(ctx, source.BaseUrl, requestMasterdata.Name)
	cacel()
	if err != nil {
		global.LOG.Error(fmt.Sprintf("从数据源 %s 获取Masterdata 失败", requestMasterdata.Source), zap.Error(err))
		response.FailWithDetailed(result, fmt.Sprintf("从数据源 %s 获取Masterdata 失败", requestMasterdata.Source), c)
		return
	}

	if len(requestMasterdata.Name) == 1 { // 如果只请求了一个数据，那就直接把这一个数据放出来
		if masterdata, ok := masterdatas[requestMasterdata.Name[0]]; ok {
			result = masterdata
		}
	}
	if result != nil { // 如果确实请求到了数据，返回这些数据
		response.OkWithData(result, c)
	} else {
		response.FailWithMessage("从所有数据源获取Masterdata失败", c)
	}
}
