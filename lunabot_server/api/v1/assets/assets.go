package assets

import (
	"context"
	"errors"
	"fmt"
	"io"
	"lunabot/xmlq/server/global"
	"lunabot/xmlq/server/model/assets/request"
	"lunabot/xmlq/server/model/common/response"
	"lunabot/xmlq/server/third_service/harukiapi"
	"lunabot/xmlq/server/third_service/unipjsk"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

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

	// 从多个源请求，只要有一个成功就返回
	type downloadFunc func(context.Context, string, string) (*http.Response, error)
	type downloadTask struct {
		SourceName string
		BaseUrl    string
		DoFunc     downloadFunc
	}
	var tasks []downloadTask

	for sourceName, source := range global.CONFIG.Assets.Sources[requestAsset.Region] {
		if source.BaseUrl == "" {
			continue
		}
		// 如果该数据源只提供某些前缀的资源，检查请求的path是否拥有相应的前缀
		if len(source.Prefixes) > 0 &&
			!slices.ContainsFunc(
				source.Prefixes,
				func(prefix string) bool {
					return strings.HasPrefix(requestAsset.Path, prefix)
				}) {
			continue
		}
		var doFunc downloadFunc
		switch sourceName {
		case harukiapi.HARUKI:
			doFunc = harukiAssetsService.DownloadAssets
		case unipjsk.UNIPJSK:
			doFunc = unipjskAssetsService.DownloadAssets
		default:
			global.LOG.Warn(fmt.Sprintf("数据源  %s 没有获取资源的方法", sourceName))
			continue
		}
		tasks = append(tasks, downloadTask{
			SourceName: sourceName,
			BaseUrl:    source.BaseUrl,
			DoFunc:     doFunc,
		})
	}
	if len(tasks) == 0 {
		c.JSON(http.StatusInternalServerError, gin.H{
			"detail": "无可用的数据源",
		})
		return
	}
	// 等待所有goroutine完成
	var wg sync.WaitGroup
	// 上下文控制超时
	ctx, cancel := context.WithTimeout(c, time.Duration(global.CONFIG.Assets.Timeout)*time.Second)
	defer cancel()
	// 用来接收响应数据
	respChan := make(chan *http.Response, len(tasks))
	for _, task := range tasks {
		wg.Add(1)
		go func(t downloadTask) {
			resp, err := t.DoFunc(ctx, t.BaseUrl, requestAsset.Path)
			if err != nil {
				if !errors.Is(err, context.Canceled) {
					global.LOG.Error(fmt.Sprintf("从数据源 %s 下载资源失败", t.SourceName), zap.Error(err))
				} else {
					global.LOG.Debug(fmt.Sprintf("从数据源 %s 下载资源时上下文被取消", t.SourceName), zap.Error(err))
				}
			} else {
				respChan <- resp
			}
			wg.Done()
		}(task)
	}
	// 等待所有任务完成后关闭channel
	go func() {
		wg.Wait()
		close(respChan)
	}()
	// 是否下载成功的标志位
	success := false
	for resp := range respChan {
		if resp == nil {
			continue
		}
		// 使用流式传输，当作它一定成功
		success = true
		contentType := resp.Header.Get("Content-Type")
		contentLength := resp.Header.Get("Content-Length")
		c.Header("Content-Type", contentType)
		if contentLength != "" {
			c.Header("Content-Length", contentLength)
		}
		c.Status(http.StatusOK)
		// 使用io.Copy进行流式传输
		_, err := io.Copy(c.Writer, resp.Body)
		resp.Body.Close()
		if err != nil {
			// 使用流式传输数据时，如果失败了就只能断开连接，不能再读取下一个源
			global.LOG.Error("流式传输数据失败", zap.Error(err))
		}
		break
	}
	// 全部失败
	if !success {
		c.JSON(http.StatusInternalServerError, gin.H{
			"detail": "从所有数据源获取解包数据失败",
		})
	}
	// 已经成功，但是可能有还在执行的goroutine，将它们的resp.Body给关掉
	go func() {
		for resp := range respChan {
			if resp != nil {
				resp.Body.Close()
			}
		}
	}()
}
