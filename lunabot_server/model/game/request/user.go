package request

type User struct {
	Region string   `json:"region" form:"region"  binding:"required"`
	UserId string   `json:"userId" form:"userId"  binding:"required"`
	Filter []string `json:"filter" form:"filter"`
}
