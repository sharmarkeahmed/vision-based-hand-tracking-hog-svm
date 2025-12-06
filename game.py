# game.py

import cv2

from config import (
    PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_X_OFFSET,
    PADDLE_ALPHA, AI_PADDLE_ALPHA,
    BALL_RADIUS, BALL_SPEED_X, BALL_SPEED_Y,
)


class PongGame:
    """
    Encapsulates Pong game state and logic:
    - Player (right) and AI (left) paddles
    - Ball position and velocity
    - Scoring (player vs AI)
    - Collisions and simple AI behavior
    """

    def __init__(self):
        self.initialized = False

        self.player_paddle_cy = None
        self.ai_paddle_cy = None

        self.ball_x = None
        self.ball_y = None
        self.ball_vx = BALL_SPEED_X
        self.ball_vy = BALL_SPEED_Y

        self.player_score = 0
        self.ai_score = 0

    def _init_if_needed(self, H, W):
        if self.initialized:
            return
        self.player_paddle_cy = H // 2
        self.ai_paddle_cy = H // 2
        self.ball_x = W // 2
        self.ball_y = H // 2
        self.initialized = True

    def update(self, H, W, dt, fingertip_y):
        """
        Update paddles and ball for one frame.

        Args:
            H, W: frame height and width.
            dt: time step (seconds).
            fingertip_y: smoothed fingertip y position in frame coords, or None.

        Returns:
            (lp_rect, rp_rect):
                lp_rect = (x1, y1, x2, y2) for left (AI) paddle
                rp_rect = (x1, y1, x2, y2) for right (player) paddle
        """
        self._init_if_needed(H, W)
        half_h = PADDLE_HEIGHT // 2

        # Player paddle follows fingertip (if present)
        if fingertip_y is not None:
            if self.player_paddle_cy is None:
                self.player_paddle_cy = fingertip_y
            else:
                self.player_paddle_cy = PADDLE_ALPHA * fingertip_y + \
                                        (1 - PADDLE_ALPHA) * self.player_paddle_cy

        if self.player_paddle_cy is None:
            self.player_paddle_cy = H // 2

        self.player_paddle_cy = max(half_h, min(H - half_h, self.player_paddle_cy))

        # Move ball
        if dt > 0:
            self.ball_x += self.ball_vx * dt
            self.ball_y += self.ball_vy * dt

        # Top/bottom walls
        if self.ball_y - BALL_RADIUS < 0:
            self.ball_y = BALL_RADIUS
            self.ball_vy *= -1
        elif self.ball_y + BALL_RADIUS > H:
            self.ball_y = H - BALL_RADIUS
            self.ball_vy *= -1

        # Paddle rectangles
        lp_x1 = PADDLE_X_OFFSET
        lp_x2 = lp_x1 + PADDLE_WIDTH
        lp_y1 = int(self.ai_paddle_cy - half_h)
        lp_y2 = int(self.ai_paddle_cy + half_h)

        rp_x2 = W - PADDLE_X_OFFSET
        rp_x1 = rp_x2 - PADDLE_WIDTH
        rp_y1 = int(self.player_paddle_cy - half_h)
        rp_y2 = int(self.player_paddle_cy + half_h)

        # AI paddle follows ball y
        ai_target_cy = self.ball_y
        self.ai_paddle_cy = (1 - AI_PADDLE_ALPHA) * self.ai_paddle_cy + \
                            AI_PADDLE_ALPHA * ai_target_cy
        self.ai_paddle_cy = max(half_h, min(H - half_h, self.ai_paddle_cy))
        lp_y1 = int(self.ai_paddle_cy - half_h)
        lp_y2 = int(self.ai_paddle_cy + half_h)

        # Collisions: right (player) paddle
        if (self.ball_x + BALL_RADIUS >= rp_x1 and
            self.ball_x - BALL_RADIUS <= rp_x2 and
            rp_y1 <= self.ball_y <= rp_y2 and
            self.ball_vx > 0):
            self.ball_x = rp_x1 - BALL_RADIUS
            self.ball_vx *= -1

        # Collisions: left (AI) paddle
        if (self.ball_x - BALL_RADIUS <= lp_x2 and
            self.ball_x + BALL_RADIUS >= lp_x1 and
            lp_y1 <= self.ball_y <= lp_y2 and
            self.ball_vx < 0):
            self.ball_x = lp_x2 + BALL_RADIUS
            self.ball_vx *= -1

        # Off-screen (scoring)
        if self.ball_x < -BALL_RADIUS:
            # Ball exited left side → YOU score
            self.player_score += 1
            self.ball_x = W // 2
            self.ball_y = H // 2
            self.ball_vx = abs(self.ball_vx)  # send it right
        elif self.ball_x > W + BALL_RADIUS:
            # Ball exited right side → AI scores
            self.ai_score += 1
            self.ball_x = W // 2
            self.ball_y = H // 2
            self.ball_vx = -abs(self.ball_vx)  # send it left

        lp_rect = (lp_x1, lp_y1, lp_x2, lp_y2)
        rp_rect = (rp_x1, rp_y1, rp_x2, rp_y2)
        return lp_rect, rp_rect

    def draw(self, display, lp_rect, rp_rect):
        """
        Draw paddles and ball onto the display frame.
        Score text and HUD are handled in main.py.
        """
        (lp_x1, lp_y1, lp_x2, lp_y2) = lp_rect
        (rp_x1, rp_y1, rp_x2, rp_y2) = rp_rect

        # AI paddle (left)
        cv2.rectangle(display, (lp_x1, lp_y1), (lp_x2, lp_y2),
                      (255, 0, 0), -1)
        cv2.putText(display, "AI",
                    (lp_x1, max(lp_y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 0), 2)

        # Player paddle (right)
        cv2.rectangle(display, (rp_x1, rp_y1), (rp_x2, rp_y2),
                      (0, 255, 0), -1)
        cv2.putText(display, "YOU",
                    (rp_x1 - 10, max(rp_y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

        # Ball
        cv2.circle(display, (int(self.ball_x), int(self.ball_y)),
                   BALL_RADIUS, (0, 255, 255), -1)
