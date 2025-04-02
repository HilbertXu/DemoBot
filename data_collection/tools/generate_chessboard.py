import cv2
import numpy as np

# Set number of inner corners (not squares!)
cols = 9  # number of inner columns (corners)
rows = 6  # number of inner rows (corners)
square_size_mm = 25  # size of one square in mm

# Total squares = inner corners + 1
image_cols = cols + 1
image_rows = rows + 1

square_size_px = 100  # pixels per square for printing

# Create the chessboard image
board = np.zeros((image_rows * square_size_px, image_cols * square_size_px), dtype=np.uint8)

for row in range(image_rows):
    for col in range(image_cols):
        if (row + col) % 2 == 0:
            x0 = col * square_size_px
            y0 = row * square_size_px
            board[y0:y0+square_size_px, x0:x0+square_size_px] = 255

# Save as image or PDF for printing
cv2.imwrite('chessboard.png', board)
print("Chessboard saved as chessboard.png")