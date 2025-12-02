package utils

import (
	"encoding/json"
	"fmt"
	"strings"
)

func GetJSONKeys(jsonStr string) (keys []string, err error) {
	// 使用json.Decoder，以便在解析过程中记录键的顺序
	dec := json.NewDecoder(strings.NewReader(jsonStr))
	t, err := dec.Token()
	if err != nil {
		return nil, err
	}
	// 确保数据是一个对象
	if t != json.Delim('{') {
		return nil, err
	}
	for dec.More() {
		t, err = dec.Token()
		if err != nil {
			return nil, err
		}
		keys = append(keys, t.(string))

		// 解析值
		var value interface{}
		err = dec.Decode(&value)
		if err != nil {
			return nil, err
		}
	}
	return keys, nil
}

// 从json数据中查找目标数据，
// 必须是使用encoding/json解析而来的值。
//
// keys只能是string或int。
// key = string时，当前的json值必须是json对象.
// key = int时，当前的json值必须是json数组.
func GetJSONValue(Json interface{}, Keys ...interface{}) (v interface{}, err error) {
	v = Json
	for i, k := range Keys {
		if v == nil {
			err = fmt.Errorf("#%d key: %v, v is nil", i, k)
			return
		}
		switch key := k.(type) {
		case int:
			if value, ok := v.([]interface{}); ok {
				if key < 0 || key >= len(value) {
					err = fmt.Errorf("#%d key: %v out of range (length = %d)", i, k, len(value))
					return
				}
				v = value[key]
			} else {
				err = fmt.Errorf("#%d key: %v type error", i, k)
				return
			}
		case string:
			if value, ok := v.(map[string]interface{}); ok {
				if v, ok = value[key]; !ok {
					err = fmt.Errorf("#%d key: %v not in map", i, k)
					return
				}
			} else {
				err = fmt.Errorf("#%d key: %v type error", i, k)
				return
			}
		default:
			err = fmt.Errorf("#%d key: %v unknown type", i, k)
			return
		}
	}
	return
}
