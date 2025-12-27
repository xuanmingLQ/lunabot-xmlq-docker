package config

type Masterdata struct {
	Timeout int                                    `mapstructure:"timeout" json:"timeout" yaml:"timeout"`
	Sources map[string]map[string]MasterdataSource `mapstructure:"sources" json:"sources" yaml:"sources"`
}

type MasterdataSource struct {
	BaseUrl    string `mapstructure:"base-url" json:"base-url" yaml:"base-url"`
	VersionUrl string `mapstructure:"version-url" json:"version-url" yaml:"version-url"`
}
