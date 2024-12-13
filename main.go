package main

import (
	"fmt"

	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/sequential"
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/tictactoe"
)

func main() {
	//trainer.TrainTicTacToe()
	//return

	Model := sequential.NewSequentialModel()
	Model.Init(9, "categoricalcrossentropy", "adam", []float64{1e-3, 0.9, 0.999, 1e-8})
	Model.AddLayer("tanh", 128)
	Model.AddLayer("tanh", 64)
	Model.AddLayer("tanh", 32)
	Model.AddLayer("softmax", 9)

	for true {
		game := tictactoe.NewTicTacToe()
		game.Init()
		game.PrintBoardMap()
		for !game.IsGameOver() {
			mapping := []string{"A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"}
			loop := true
			var input string
			for loop {
				fmt.Print("You Playing : ")
				fmt.Scan(&input)
				loop = game.Move(input)
			}
			game.PrintBoard()
			game.ChangePlayer()
			fmt.Println("\nAI Playing")
			index := tictactoe.AIPlaying(Model, game)
			game.Move(mapping[index])
			game.PrintBoard()
			fmt.Println()
			game.ChangePlayer()
		}
	}
}
