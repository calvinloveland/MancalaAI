import unittest

from gym_mancala.envs.board import Board


class TestEnv(unittest.TestCase):

    def test_board_initialization(self):
        board = Board()
        for i in range(1):
            for j in range(6):
                assert board.marbles[i][j] == 4
        board.execute_turn(1)
        new_board = Board(board)
        board.execute_turn(1)
        board.execute_turn(1)
        assert not new_board.game_over





if __name__ == '__main__':
    print("Note:")
    print("These tests mainly test the Gym environment.")
    print("If you would like to see networks train or test networks please run __main__.py.", flush=True)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnv)
    unittest.TextTestRunner(verbosity=2).run(suite)
