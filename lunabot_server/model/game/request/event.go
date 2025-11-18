package request

type Event struct {
	Region  string `json:"region" form:"region"  binding:"required"`
	EventId string `json:"eventId" form:"eventId"  binding:"required"`
}
