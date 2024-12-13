package trainer

import (
	"fmt"

	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/loss"
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/sequential"
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/tictactoe"
)

func AIPlaying(model *sequential.SequentialModel, states *[][]float64, targets *[][]float64) {
	mapping := []string{"A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"}
	board := tictactoe.NewTicTacToe()
	board.Init()
	for !board.IsGameOver() {
		boardInput := board.Flatten()
		valid := board.ValidMove()
		target := make([]float64, 9)

		for i := range len(valid) {
			target[valid[i]] = 1.0 / float64(len(valid))
		}

		*states = append(*states, boardInput)
		*targets = append(*targets, target)

		index := tictactoe.AIPlaying(model, board)
		board.Move(mapping[index])
		board.ChangePlayer()
	}
}

func TrainTicTacToe() {
	Model := sequential.NewSequentialModel()
	Model.Init(9, "mse", "adam", []float64{1e-3, 0.9, 0.999, 1e-8})

	Model.AddLayer("relu", 128)
	Model.AddLayer("relu", 64)
	Model.AddLayer("sigmoid", 9)

	for epoch := range 1000 {
		var states [][]float64 = [][]float64{}
		var targets [][]float64 = [][]float64{}

		AIPlaying(Model, &states, &targets)

		for i := range len(states) {
			Model.Predict(states[i])
			Model.Backward(targets[i])
		}

		lossVal := 0.0
		for i := range len(states) {
			output := Model.Predict(states[i])
			lossVal += loss.MSEError(output, targets[i])
		}

		lossVal = lossVal / float64(len(states))

		if epoch%100 == 0 {
			fmt.Printf("Loss : %f \n", lossVal)
		}
	}

	game := tictactoe.NewTicTacToe()
	game.Init()
	for !game.IsGameOver() {
		mapping := []string{"A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"}

		index := tictactoe.AIPlaying(Model, game)
		game.PrintBoard()
		game.Move(mapping[index])
		game.ChangePlayer()
		fmt.Println()
	}
	game.PrintBoard()
	Model.Export("./models/tictactoe.json")
}