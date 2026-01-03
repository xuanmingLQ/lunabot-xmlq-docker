package response

import "time"

type UploadTime struct {
	UploadTime *time.Time `json:"upload_time"` // 这里是为了和其他的 upload_time一致
	Error      error      `json:"error"`
}
