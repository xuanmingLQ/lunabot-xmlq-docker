package config

type Server struct {
	Zap      Zap        `mapstructure:"zap" json:"zap" yaml:"zap"`
	Upload   Upload     `mapstructure:"upload" json:"upload" yaml:"upload"`
	DiskList []DiskList `mapstructure:"disk-list" json:"disk-list" yaml:"disk-list"`
	System   System     `mapstructure:"system" json:"system" yaml:"system"`
	// 数据库
	Redis Redis `mapstructure:"redis" json:"redis" yaml:"redis"`
	Pgsql Pgsql `mapstructure:"pgsql" json:"pgsql" yaml:"pgsql"`
	// 第三方服务
	Masterdata Masterdata `mapstructure:"masterdata" json:"masterdata" yaml:"masterdata"`
	Assets     Assets     `mapstructure:"assets" json:"assets" yaml:"assets"`
	HarukiApi  HarukiApi  `mapstructure:"haruki-api" json:"haruki-api" yaml:"haruki-api"`
}
