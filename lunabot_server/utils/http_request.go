package utils

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
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
	Status int    
	Url    string 
	Detail string 
}

func (he *HttpError) Error() string {
	if he == nil {
		return "未知错误"
	}
	return fmt.Sprintf("%d: %s", he.Status, he.Detail)
}

var HTTP_ERROR = &HttpError{}

func (*HttpError) Is(err error) bool {
	return err == HTTP_ERROR
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
				// 整个连接池对所有主机的最大空闲连接数
				MaxIdleConns: 100,
				// 每个目标主机（Host）保持的最大空闲连接数
				MaxIdleConnsPerHost: 20,
				// 连接空闲多久后关闭
				IdleConnTimeout: 90 * time.Second,
				// 握手超时
				TLSHandshakeTimeout: 30 * time.Second,
				// 跳过tls验证， 危险，可能会受到中间人攻击
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
		httpError := HttpError{
			Status: resp.StatusCode,
			Url:    Req.URL.String(),
		}
		detail, _ := io.ReadAll(resp.Body)
		httpError.Detail = string(detail)
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
