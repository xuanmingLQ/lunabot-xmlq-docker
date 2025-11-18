package request

type Asset struct {
	Region string `json:"region" form:"region" uri:"region" binding:"required"`
	Path   string `json:"path" form:"path" uri:"path" binding:"required"`
}
