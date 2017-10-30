import pygame # Create and interact with GUI for games in Python
import random

# Hyper Parameters
# Frame rate
FPS = 60

# Window size
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400

# Paddle size
PADDLE_WIDTH = 60
PADDLE_HEIGHT = 10

# Distance of paddle from edge of window
PADDLE_BUFFER = 10

# Ball size
BALL_WIDTH = 10
BALL_HEIGHT = 10

# Speeds of ball and paddle
PADDLE_SPEED = 3
BALL_X_SPEED = 2
BALL_Y_SPEED = 2

# RGB colors for ball, paddle, and background
WHITE = (255, 255, 255) # Colour of ball and paddle
BLACK = (0, 0, 0) # Colour of background

# Initialize screen
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

# Method for resetting positions once game is over
def reset_positions():
    # Reset paddle to middle of window
    paddle1XPos = (WINDOW_WIDTH - 10) / 2

    # Random integers used to vary initia position of ball
    num_1 = random.randint(-100, 100)
    num_2 = random.randint(-100, 100)

    # Reposition ball near center of screen, with small random variation in height and position relative to agent
    ballXPos, ballYPos = (WINDOW_WIDTH / 2 + num_1), (WINDOW_HEIGHT /2 + num_2)

    num_3 = random.randint(0,11) # Random number 0 <= num_3 <= 11

    # Use random number to decide initial direction of ball
    if (0 <= num_3 < 3):
        ballXDirection, ballYDirection = 1, 1
    elif (3 <= num_3 < 6):
        ballXDirection, ballYDirection = -1, 1
    elif (6 <= num_3 < 9):
        ballXDirection, ballYDirection = 1, -1
    else:
        ballXDirection, ballYDirection = -1, -1

    return [paddle1XPos, ballXPos, ballYPos, ballXDirection, ballYDirection]

def drawBall(ballXPos, ballYPos):
    # Create small rectangle representing pong ball
    ball = pygame.Rect(ballXPos, ballYPos, BALL_WIDTH, BALL_HEIGHT)
    # Draw ball
    pygame.draw.rect(screen, WHITE, ball)

# Name it paddle1 so we can insert a second paddle to play the AI later
def drawPaddle1(paddle1XPos):
    # Create paddle on bottom face
    paddle1 = pygame.Rect(paddle1XPos, WINDOW_HEIGHT - (PADDLE_BUFFER + PADDLE_HEIGHT), PADDLE_WIDTH,
                          PADDLE_HEIGHT)  # Use buffer to avoid paddle leaving the screen
    # Draw paddle1
    pygame.draw.rect(screen, WHITE, paddle1)

# Update the ball - using positions and directions of ball and paddle - and allocate scores
def updateFrame(paddle1XPos, ballXPos, ballYPos, ballXDirection, ballYDirection):

    # Initialise score
    score = 0
    # Update position of ball
    ballXPos = ballXPos + ballXDirection * BALL_X_SPEED
    ballYPos = ballYPos + ballYDirection * BALL_Y_SPEED

    # If the ball hits the top, reflect it downward
    if ballYPos <= 0:
        ballYDirection = 1

    # If the ball hits the (bottom) paddle, reflect it upward, and give the agent a reward
    if (ballYPos >= WINDOW_HEIGHT - (PADDLE_HEIGHT + PADDLE_BUFFER) and ballXPos + BALL_WIDTH >= paddle1XPos and ballXPos <= paddle1XPos + PADDLE_WIDTH):
        ballYDirection = -1

        score = 1
    elif (ballYPos >= WINDOW_HEIGHT - BALL_HEIGHT): #else, give agent a penalty and reset positions for ball and paddle
        score = -1

        [paddle1XPos, ballXPos, ballYPos, ballXDirection, ballYDirection] = reset_positions()

        return [score, paddle1XPos, ballXPos, ballYPos, ballXDirection, ballYDirection]

    # If the ball hits the left side, reflect it back
    if (ballXPos <= 0):
        ballXPos = 0
        ballXDirection = 1
    elif (ballXPos >= WINDOW_WIDTH - BALL_WIDTH): #else, if it hits the right side, reflect back to left
        ballXPos = WINDOW_WIDTH - BALL_WIDTH
        ballXDirection = -1

    return [score, paddle1XPos, ballXPos, ballYPos, ballXDirection, ballYDirection]

# Update the paddle position
def updatePaddle1(action, paddle1XPos):
    # If the agent decides to move left...
    if (action[1] == 1):
        paddle1XPos = paddle1XPos - PADDLE_SPEED
        # If the agent decides to move right...
    if (action[2] == 1):
        paddle1XPos = paddle1XPos + PADDLE_SPEED

    # Prevent paddle from moving off the screen
    if (paddle1XPos < 0):
        paddle1XPos = 0
    if (paddle1XPos > WINDOW_WIDTH - PADDLE_WIDTH):
        paddle1XPos = WINDOW_WIDTH - PADDLE_WIDTH
    return paddle1XPos

def drawScore(score):
    font = pygame.font.Font(None, 28)
    scorelabel = font.render("Score " + str(score), 1, WHITE)
    screen.blit(scorelabel, (30 , 10))

# Define game class
class PongGame:
    def __init__(self):
        pygame.font.init()
        # Random number for initial direction of ball
        num_0 = random.randint(0,9)
        # Create tally to keep track of total score
        self.tally = 0
        # Initialise position of paddle
        self.paddle1XPos = (WINDOW_HEIGHT - PADDLE_HEIGHT) / 2

        self.ballYPos = (num_0/9)*(WINDOW_HEIGHT - BALL_HEIGHT)

        num_1 = random.randint(-50, 50)
        num_2 = random.randint(-100, 100)

        # Reposition ball near center of screen, with small random variation in height and position relative to agent
        self.ballXPos, self.ballYPos = (WINDOW_WIDTH / 2 + num_1), (WINDOW_HEIGHT / 2 + num_2)

        num_3 = random.randint(0, 11)  # Random number 0 <= num_3 <= 11

        # Use random number to decide initial direction of ball
        if (0 <= num_3 < 3):
            self.ballXDirection, self.ballYDirection = 1, 1
        elif (3 <= num_3 < 6):
            self.ballXDirection, self.ballYDirection = -1, 1
        elif (6 <= num_3 < 9):
            self.ballXDirection, self.ballYDirection = 1, -1
        else:
            self.ballXDirection, self.ballYDirection = -1, -1

    def getPresentFrame(self):
        pygame.event.pump()
        # Fill in black background
        screen.fill(BLACK)
        # Draw the paddle
        drawPaddle1(self.paddle1XPos)
        #drawPaddle2(self.paddle2XPos)
        # Draw the ball
        drawBall(self.ballXPos, self.ballYPos)
        # Print the score on screen
        drawScore(self.tally)
        # Copy the pixels from screen surface to a 3D array
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        # Update display
        pygame.display.flip()
        # Return image data
        return image_data

    #update our screen
    def getNextFrame(self, action):
        pygame.event.pump()
        score = 0
        screen.fill(BLACK)
        #update our paddle
        self.paddle1XPos = updatePaddle1(action, self.paddle1XPos)
        drawPaddle1(self.paddle1XPos)
        # Update positions and score
        [score, self.paddle1XPos, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection] = updateFrame(self.paddle1XPos, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection)
        # Draw the ball
        drawBall(self.ballXPos, self.ballYPos)
        # Get image data
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        # Update the window
        pygame.display.flip()
        # Record the total score
        self.tally = self.tally + score
        print("Tally is " + str(self.tally))
        # Return the score and the surface data
        return [score, image_data]
