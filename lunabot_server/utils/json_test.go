package utils

import (
	"encoding/json"
	"log"
	"testing"
)

/*
	func TestGetJSONKeys(t *testing.T) {
		var jsonStr = `
		{
			"Name": "test",
			"TableName": "test",
			"TemplateID": "test",
			"TemplateInfo": "test",
			"Limit": 0
	}`

		keys, err := GetJSONKeys(jsonStr)
		if err != nil {
			t.Errorf("GetJSONKeys failed" + err.Error())
			return
		}
		if len(keys) != 5 {
			t.Errorf("GetJSONKeys failed" + err.Error())
			return
		}
		if keys[0] != "Name" {
			t.Errorf("GetJSONKeys failed" + err.Error())

			return
		}
		if keys[1] != "TableName" {
			t.Errorf("%s", "GetJSONKeys failed" + err.Error())

			return
		}
		if keys[2] != "TemplateID" {
			t.Errorf("GetJSONKeys failed" + err.Error())

			return
		}
		if keys[3] != "TemplateInfo" {
			t.Errorf("GetJSONKeys failed" + err.Error())

			return
		}
		if keys[4] != "Limit" {
			t.Errorf("GetJSONKeys failed" + err.Error())

			return
		}

		fmt.Println(keys)
	}
*/
func TestGetJSONValue(t *testing.T) {
	var m1 interface{}
	var s1 interface{}
	json.Unmarshal([]byte(
		`{
			"a": [1,2,3]
	}`),
		&m1)
	json.Unmarshal([]byte(
		`[
			{ "1": {"c":"d"} }
	]`),
		&s1)
	v, err := GetJSONValue(m1, "a", 0)
	if err != nil {
		t.Errorf("get json value failed %v", err)
		return
	}
	log.Println(v)
	v, err = GetJSONValue(s1, 0, "1", "c")
	if err != nil {
		t.Errorf("get json value failed %v", err)
		return
	}
	log.Println(v)
}
