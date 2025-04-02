import cv2
import cv2.aruco as aruco

# Parameters
squares_x = 5  # number of chessboard squares in X
squares_y = 7  # number of chessboard squares in Y
square_length = 0.04  # in meters
marker_length = 0.02  # in meters

# Dictionary and board
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
board = aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, aruco_dict)

# Draw the board
img = board.draw((1000, 1400))  # size in pixels
cv2.imwrite("charuco_board.png", img)
