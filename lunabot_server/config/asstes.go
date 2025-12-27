package config

type Assets struct {
	Timeout          int                                `mapstructure:"timeout" json:"timeout" yaml:"timeout"`
	ImgCacheMaxRes   int                                `mapstructure:"img-cache-max-res" json:"img-cache-max-res" yaml:"img-cache-max-res"`
	Sources          map[string]map[string]AssetsSource `mapstructure:"sources" json:"sources" yaml:"sources"`
	OndemandPrefixes []string                           `mapstructure:"ondemand-prefixes" json:"ondemand-prefixes" yaml:"ondemand-prefixes"`
	StartAppPrefixes []string                           `mapstructure:"startapp-prefixes" json:"startapp-prefixes" yaml:"startapp-prefixes"`
}

type AssetsSource struct {
	BaseUrl  string   `mapstructure:"base-url" json:"base-url" yaml:"base-url"`
	Prefixes []string `mapstructure:"prefixes" json:"prefixes" yaml:"prefixes"`
}
