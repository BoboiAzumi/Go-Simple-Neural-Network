package tictactoe

import (
	"fmt"
	"strings"

	"github.com/BoboiAzumi/Go-Simple-Neural-Network/network/sequential"
	"github.com/BoboiAzumi/Go-Simple-Neural-Network/utils"
)

type TicTacToe struct {
	board  [][]int
	player int
}

func (game *TicTacToe) Init() {
	game.board = make([][]int, 3)
	for i := range len(game.board) {
		game.board[i] = make([]int, 3)
	}
	game.player = 1
}

func (game *TicTacToe) GameBoard() [][]int {
	return game.board
}

func (game *TicTacToe) Flatten() []float64 {
	boardFloat64 := make([][]float64, 3)
	for i := range len(boardFloat64) {
		boardFloat64[i] = make([]float64, 3)
		for j := range len(boardFloat64[i]) {
			boardFloat64[i][j] = float64(game.board[i][j])
		}
	}

	return utils.Flatten(boardFloat64)
}

func (game *TicTacToe) Move(key string) bool {
	error := false
	switch strings.ToUpper(key) {
	case "A1":
		if game.board[0][0] == 0 {
			game.board[0][0] = game.player
		} else {
			error = !error
		}
		break
	case "A2":
		if game.board[0][1] == 0 {
			game.board[0][1] = game.player
		} else {
			error = !error
		}
		break
	case "A3":
		if game.board[0][2] == 0 {
			game.board[0][2] = game.player
		} else {
			error = !error
		}
		break
	case "B1":
		if game.board[1][0] == 0 {
			game.board[1][0] = game.player
		} else {
			error = !error
		}
		break
	case "B2":
		if game.board[1][1] == 0 {
			game.board[1][1] = game.player
		} else {
			error = !error
		}
		break
	case "B3":
		if game.board[1][2] == 0 {
			game.board[1][2] = game.player
		} else {
			error = !error
		}
		break
	case "C1":
		if game.board[2][0] == 0 {
			game.board[2][0] = game.player
		} else {
			error = !error
		}
		break
	case "C2":
		if game.board[2][1] == 0 {
			game.board[2][1] = game.player
		} else {
			error = !error
		}
		break
	case "C3":
		if game.board[2][2] == 0 {
			game.board[2][2] = game.player
		} else {
			error = !error
		}
		break
	default:
		error = !error
	}

	return error
}

func (game *TicTacToe) IsGameOver() bool {
	gameover := false

	for i := range len(game.board) {
		if game.board[i][0] == game.board[i][1] && game.board[i][1] == game.board[i][2] && game.board[i][0] != 0 {
			gameover = true
		}
	}

	for i := range len(game.board) {
		for j := range len(game.board[i]) {
			if game.board[0][j] == game.board[1][j] && game.board[1][j] == game.board[2][j] && game.board[i][j] != 0 {
				gameover = true
			}
		}
	}

	if game.board[1][1] == game.board[0][0] && game.board[2][2] == game.board[0][0] && game.board[0][0] != 0 {
		gameover = true
	}
	if game.board[1][1] == game.board[0][2] && game.board[2][0] == game.board[0][2] && game.board[0][2] != 0 {
		gameover = true
	}

	count := 0
	for i := range len(game.board) {
		for j := range len(game.board) {
			if game.board[i][j] != 0 {
				count += 1
			}
		}
	}
	if count == 9 {
		gameover = true
	}

	return gameover
}

func (game *TicTacToe) PrintBoard() {
	for i := range len(game.board) {
		fmt.Printf("%d \n", game.board[i])
	}
}

func (game *TicTacToe) ChangePlayer() {
	game.player = 3 - game.player
}

func (game *TicTacToe) ValidMove() []int {
	boardFlatten := game.Flatten()
	var index []int = []int{}
	for i := range len(boardFlatten) {
		if boardFlatten[i] == 0.0 {
			index = append(index, i)
		}
	}

	return index
}

func (game TicTacToe) PrintBoardMap() {
	boardMap := make([][]string, 3)
	for i := range len(boardMap) {
		boardMap[i] = make([]string, 3)
	}
	boardMap[0][0] = "A1"
	boardMap[0][1] = "A2"
	boardMap[0][2] = "A3"
	boardMap[1][0] = "B1"
	boardMap[1][1] = "B2"
	boardMap[1][2] = "B3"
	boardMap[2][0] = "C1"
	boardMap[2][1] = "C2"
	boardMap[2][2] = "C3"

	for i := range len(boardMap) {
		fmt.Printf("%s \n", boardMap[i])
	}
}

func AIPlaying(model *sequential.SequentialModel, board *TicTacToe) int {
	boardInput := board.Flatten()
	valid := board.ValidMove()
	prediction := model.Predict(boardInput)

	best := 0.0
	index := 0
	for i := range len(valid) {
		if prediction[valid[i]] > best {
			index = valid[i]
			best = prediction[valid[i]]
		}
	}

	return index
}

func NewTicTacToe() *TicTacToe {
	return &TicTacToe{}
}
