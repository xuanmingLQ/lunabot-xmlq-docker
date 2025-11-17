package request

type Masterdata struct {
	Region string   `json:"region" form:"region"  binding:"required"`
	Source string   `json:"source" form:"source"  binding:"required"`
	Name   []string `json:"name" form:"name"  binding:"required"`
}
