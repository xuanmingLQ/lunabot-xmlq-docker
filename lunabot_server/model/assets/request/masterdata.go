package request

import (
	"errors"
	"strings"

	"github.com/gin-gonic/gin"
)

type Masterdata struct {
	Region string   `json:"region" form:"region"  binding:"required"`
	Source string   `json:"source" form:"source"  binding:"required"`
	Name   []string `json:"name" form:"name"  binding:"required"`
}

func (md *Masterdata) BindQuery(c *gin.Context) error {
	if err := c.ShouldBindQuery(md); err != nil {
		return err
	}
	if len(md.Name) == 0 {
		return errors.New("缺少参数 name")
	}
	var names []string
	for _, n := range md.Name {
		if n == "" {
			continue
		}
		for name := range strings.SplitSeq(n, ",") {
			name = strings.TrimSpace(name)
			if name == "" {
				continue
			}
			names = append(names, name)
		}
	}
	md.Name = names
	return nil
}
