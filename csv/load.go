package csv

import (
	"encoding/csv"
	"os"
)

type CSV struct {
	Column []string
	Data   [][]string
}

func Load(path string) *CSV {
	file, err := os.Open(path)

	if err != nil {
		panic("File not found")
	}

	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()

	if err != nil {
		panic("Error parsing CSV")
	}

	var CSV_DATA CSV = CSV{
		Column: records[0],
		Data:   records[1:],
	}

	return &CSV_DATA
}
