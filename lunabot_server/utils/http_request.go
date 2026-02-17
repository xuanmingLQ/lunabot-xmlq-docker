package utils

import (
	"compress/gzip"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/dsnet/compress/brotli"
	"github.com/klauspost/compress/zstd"
)

const (
	EncodingGzip = "gzip"
	EncodingBr   = "br"
	EncodingZstd = "zstd"
)

var (
	hc *http.Client = &http.Client{
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
)

type HttpError struct {
	StatusCode int
	Url        string
	Detail     string
}

func (he *HttpError) Error() string {
	if he == nil {
		return "未知错误"
	}
	return fmt.Sprintf("访问 %s 异常：%d %s", he.Url, he.StatusCode, he.Detail)
}

var HTTP_ERROR = &HttpError{}

func (*HttpError) Is(target error) bool {
	_, ok := target.(*HttpError)
	return ok
}

type HttpResult struct {
	Resp  *http.Response
	Body  io.ReadCloser
	Error *HttpError
}

// 关闭body
func (r *HttpResult) Close() (err error) {
	if r.Body != nil {
		err = r.Body.Close()
		r.Body = nil
	}
	if r.Resp != nil && r.Resp.Body != nil {
		err = errors.Join(err, r.Resp.Body.Close())
		r.Resp.Body = nil
	}
	return
}

// 将body读取为json，并关闭
func (r *HttpResult) Json(v interface{}) error {
	if r.Body == nil {
		return errors.New("内容为空")
	}
	decoder := json.NewDecoder(r.Body)
	decoder.UseNumber()
	err := decoder.Decode(&v)
	r.Close()
	return err
}

// 将body读取为字节数组，并关闭
func (r *HttpResult) Bytes() (b []byte, err error) {
	if r.Body == nil {
		err = errors.New("内容为空")
	}
	b, err = io.ReadAll(r.Body)
	r.Close()
	return
}

func HttpRequest(Req *http.Request) (result *HttpResult) {
	result = &HttpResult{}
	Url := Req.URL.String()
	resp, err := hc.Do(Req)
	if err != nil {
		result.Error = &HttpError{
			Url:    Url,
			Detail: err.Error(),
		}
		return
	}
	result.Resp = resp
	switch strings.ToLower(resp.Header.Get("content-encoding")) {
	case EncodingGzip:
		result.Body, err = gzip.NewReader(resp.Body)
	case EncodingBr:
		result.Body, err = brotli.NewReader(resp.Body, &brotli.ReaderConfig{})
	case EncodingZstd:
		result.Body, err = NewZstdReadCloser(resp.Body)
	default:
		result.Body = resp.Body
	}
	if err != nil {
		result.Close()
		result.Error = &HttpError{
			Url:        Url,
			StatusCode: resp.StatusCode,
			Detail:     err.Error(),
		}
		return
	}
	if resp.StatusCode != http.StatusOK {
		defer result.Close()
		result.Error = &HttpError{
			StatusCode: resp.StatusCode,
			Url:        Url,
		}
		body, _ := io.ReadAll(io.LimitReader(result.Body, 1024*10))
		if len(body) == 0 {
			return
		} else if strings.Contains(resp.Header.Get("Content-Type"), "application/json") {
			// json格式，全部保存
			result.Error.Detail = string(body)
		} else {
			// 其它类型，保存前200字符
			detail := string(body)
			if len(detail) > 200 {
				result.Error.Detail = detail[:200] + "..."
			} else {
				result.Error.Detail = detail
			}
		}
		return
	}
	return
}

// 由于zstd.Decoder的Close不会返回error，导致它没有实现io.Closer接口
// 用自定义类型包装一下，实现io.Closer接口
type zstdReadClose struct {
	*zstd.Decoder
}

func NewZstdReadCloser(r io.Reader) (*zstdReadClose, error) {
	d, err := zstd.NewReader(r)
	if err != nil {
		return nil, err
	}
	return &zstdReadClose{d}, nil
}

func (z *zstdReadClose) Close() error {
	z.Decoder.Close()
	return nil
}
