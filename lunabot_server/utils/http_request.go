package utils

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const (
	DataTypeNone = iota
	DataTypeJson
	DataTypeBytes
)

var (
	hc *http.Client
)

type HttpError struct {
	Status  int    `json:"status"`
	Message string `json:"message"`
}

func (he *HttpError) Error() string {
	if he == nil {
		return "未知错误"
	}
	return fmt.Sprintf("%d: %s", he.Status, he.Message)
}

// DataTypeNone 返回Response
// DataTypeJson 返回json.Decode的结果，其中的number会被解析为json.Number类型
// DataTypeBytes 返回[]byte
func HttpRequest(
	Req *http.Request,
	DataType int) (interface{}, error) {
	if hc == nil {
		hc = &http.Client{
			Transport: &http.Transport{
				TLSClientConfig: &tls.Config{
					InsecureSkipVerify: true,
				},
			},
		}
	}
	resp, err := hc.Do(Req)
	if err != nil {
		return nil, err
	}
	// 如果不需要原始响应，将它关闭
	if DataType != DataTypeNone {
		defer resp.Body.Close()
	}
	if resp.StatusCode != http.StatusOK {
		var httpError HttpError
		_ = json.NewDecoder(resp.Body).Decode(&httpError)
		if httpError.Status == 0 {
			httpError.Status = resp.StatusCode
		}
		return nil, &httpError
	}
	switch DataType {
	case DataTypeNone:
		return resp, err
	case DataTypeJson:
		// 解码jsons
		var result interface{}
		decoder := json.NewDecoder(resp.Body)
		decoder.UseNumber()
		err = decoder.Decode(&result)
		return result, err
	case DataTypeBytes:
		// 把即将关闭的body读出
		result, err := io.ReadAll(resp.Body)
		return result, err
	default:
		return nil, fmt.Errorf("不支持的数据类型： %d", DataType)
	}
}
