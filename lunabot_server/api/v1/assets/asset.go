package assets

import (
	"fmt"
	"io"
	"lunabot/xmlq/server/global"
	"lunabot/xmlq/server/model/assets/request"
	"lunabot/xmlq/server/model/common/response"
	"net/http"
	"slices"
	"strings"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

type AssetApi struct{}

// DownloadRipAssets
// @Summary 下载解包资源
// @Produce application/json
// @Param data query request.Rip
// @Success 200 {object}
// @Router /asset/downloadAsset [get]
func (*AssetApi) DownloadAsset(c *gin.Context) {
	var requestAsset request.Asset
	if err := c.ShouldBindUri(&requestAsset); err != nil {
		global.LOG.Error("参数校验不通过！", zap.Error(err))
		response.FailWithMessage("参数校验不通过", c)
		return
	}

	for sourceName, ripSource := range global.CONFIG.SekaiAsset.Sources[requestAsset.Region].Asset {
		if ripSource.BaseUrl == "" {
			continue
		}
		// 如果该数据源只提供某些前缀的资源，检查请求的path是否拥有相应的前缀
		if len(ripSource.Prefixes) > 0 &&
			!slices.ContainsFunc(ripSource.Prefixes, func(prefix string) bool { return strings.HasPrefix(requestAsset.Path, prefix) }) {
			continue
		}
		var resp *http.Response
		var err error
		switch sourceName {
		case HARUKI:
			resp, err = harukiAssetService.DownloadAsset(c, ripSource.BaseUrl, requestAsset.Path)
		default:
			global.LOG.Warn(fmt.Sprintf("数据源: %s 没有指定获取方法", sourceName))
			continue
		}
		if err != nil {
			global.LOG.Error(fmt.Sprintf("从数据源：%s 获取解包数据：%s 失败", sourceName, requestAsset.Path), zap.Error(err))
			continue
		}
		contentType := resp.Header.Get("Content-Type")
		resultBody, err := io.ReadAll(resp.Body)
		//
		resp.Body.Close()
		if err != nil {
			global.LOG.Error("解析响应体数据失败", zap.Error(err))
			continue
		}
		// 只需要从一个数据源获取数据就好
		c.Data(http.StatusOK, contentType, resultBody)
		return
	}
	// 所有数据源均获取失败
	c.JSON(http.StatusNotFound, gin.H{
		"detail": "从所有数据源获取解包数据失败",
	})
}
