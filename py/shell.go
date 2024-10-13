package py

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
)

func CalculateAreturnB(scriptName string, op string, input []float64, shape []int) (result []float64, resultShape []int, err error) {
	inputJSON, err := json.Marshal(input)
	if err != nil {
		return
	}
	shapeJSON, err := json.Marshal(shape)
	if err != nil {
		return
	}
	cmd := exec.Command("python3", scriptName, op, string(inputJSON), string(shapeJSON))
	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println("shell output:", string(output))
		return
	}

	// 解析Python返回的结果
	var outputLines []string = strings.Split(string(output), "\n")
	err = json.Unmarshal([]byte(outputLines[0]), &result)
	if err != nil {
		return
	}
	err = json.Unmarshal([]byte(outputLines[1]), &resultShape)
	return
}
func CalculateA_breturnC(scriptName string, op string, a []float64, shapeA []int, b []int) (c []float64, resultShape []int, err error) {
	aJSON, err := json.Marshal(a)
	if err != nil {
		return
	}
	shapeAJSON, err := json.Marshal(shapeA)
	if err != nil {
		return
	}
	bJSON, err := json.Marshal(b)
	if err != nil {
		return
	}
	cmd := exec.Command("python3", scriptName, op, string(aJSON), string(shapeAJSON), string(bJSON))
	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println(string(output))
		return
	}

	// 解析Python返回的结果
	var outputLines []string = strings.Split(string(output), "\n")
	err = json.Unmarshal([]byte(outputLines[0]), &c)
	if err != nil {
		return
	}
	err = json.Unmarshal([]byte(outputLines[1]), &resultShape)
	return
}

// 调用Python脚本计算预期结果
func CalculateA_B_ReturnC(scriptName string, op string, a, b []float64, shapeA, shapeB []int) (c []float64, resultShape []int, err error) {
	aJSON, err := json.Marshal(a)
	if err != nil {
		return
	}
	bJSON, err := json.Marshal(b)
	if err != nil {
		return
	}
	shapeAJSON, err := json.Marshal(shapeA)
	if err != nil {
		return
	}
	shapeBJSON, err := json.Marshal(shapeB)
	if err != nil {
		return
	}
	cmd := exec.Command("python3", scriptName, op, string(aJSON), string(shapeAJSON), string(bJSON), string(shapeBJSON))
	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println(string(output))
		return
	}

	// 解析Python返回的结果
	var result []string = strings.Split(string(output), "\n")
	result[1] = strings.Replace(result[1], "(", "[", -1)
	result[1] = strings.Replace(result[1], ")", "]", -1)
	// fmt.Println(result[0])
	// fmt.Println(result[1])

	err = json.Unmarshal([]byte(result[0]), &c)
	if err != nil {
		return
	}
	err = json.Unmarshal([]byte(result[1]), &resultShape)
	return
}
