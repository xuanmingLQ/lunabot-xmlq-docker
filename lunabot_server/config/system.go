package config

type System struct {
	RouterPrefix string `mapstructure:"router-prefix" json:"router-prefix" yaml:"router-prefix"`
	Addr         int    `mapstructure:"addr" json:"addr" yaml:"addr"`                // 端口值
	UseRedis     bool   `mapstructure:"use-redis" json:"use-redis" yaml:"use-redis"` // 使用redis
	Root         string `mapstructure:"root" json:"root" yaml:"root"`
}
