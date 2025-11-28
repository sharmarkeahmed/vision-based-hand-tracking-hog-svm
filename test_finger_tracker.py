import cv2
import numpy as np
import time
from joblib import load

from hog_utils import create_hog, compute_hog, sliding_windows

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

HOG_WIN_SIZE    = (64, 64)
BLOCK_SIZE      = (16, 16)
BLOCK_STRIDE    = (8, 8)
CELL_SIZE       = (8, 8)
NBINS           = 9

SVM_MODEL_PATH  = "models/hog_finger_svm.joblib"

SLIDE_STEP      = 32      # pixels between windows on the small frame
DETECT_SCALE    = 0.5     # run HOG on downscaled frame for speed
SCORE_THRESHOLD = 0.0     # minimum SVM score to accept detection

DETECT_EVERY_N  = 3       # do full HOG detection every N frames

# Exclusion zone (middle vertical band in the DOWNSCALED frame)
EXCLUDE_X1_FRAC = 0.30    # left edge of middle band (30% of width)
EXCLUDE_X2_FRAC = 0.70    # right edge of middle band (70% of width)
EXCLUDE_Y1_FRAC = 0.0
EXCLUDE_Y2_FRAC = 1.0

# Paddle config (vertical paddle at left or right)
PADDLE_WIDTH    = 20
PADDLE_HEIGHT   = 120
PADDLE_X_OFFSET = 40      # distance from each side wall
PADDLE_ALPHA    = 0.4     # smoothing factor for paddle movement (Y)


# ---------------------------------------------------------
# FINGER DETECTION (HOG + SVM)
# ---------------------------------------------------------

def detect_finger_hog(frame, hog, clf):
    """
    Run sliding-window HOG + SVM on a downscaled frame.
    Ignore windows whose center is inside the central vertical band
    (where your face is likely to be). Only use left/right side regions.
    Returns (x, y, w, h), best_score in original-frame coordinates.
    """
    small = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
    Hs, Ws = small.shape[:2]

    best_score = -np.inf
    best_box_small = None

    win_w, win_h = hog.winSize

    ex_x1 = int(EXCLUDE_X1_FRAC * Ws)
    ex_x2 = int(EXCLUDE_X2_FRAC * Ws)
    ex_y1 = int(EXCLUDE_Y1_FRAC * Hs)
    ex_y2 = int(EXCLUDE_Y2_FRAC * Hs)

    for (x, y, patch) in sliding_windows(small, hog=hog, step=SLIDE_STEP):
        cx = x + win_w // 2
        cy = y + win_h // 2

        # Skip if center in exclusion zone (middle band)
        if ex_x1 <= cx <= ex_x2 and ex_y1 <= cy <= ex_y2:
            continue

        feat = compute_hog(patch, hog)
        score = clf.decision_function([feat])[0]

        if score > best_score:
            best_score = score
            best_box_small = (x, y, win_w, win_h)

    if best_box_small is None or best_score < SCORE_THRESHOLD:
        return None, best_score

    xs, ys, ws, hs = best_box_small
    x = int(xs / DETECT_SCALE)
    y = int(ys / DETECT_SCALE)
    w = int(ws / DETECT_SCALE)
    h = int(hs / DETECT_SCALE)

    return (x, y, w, h), best_score


# ---------------------------------------------------------
# FINGERTIP ESTIMATION USING CONTOURS
# ---------------------------------------------------------

def find_fingertip_in_box(frame, hand_box):
    x, y, w, h = hand_box
    roi = frame[y:y+h, x:x+w]

    if roi.size == 0:
        return None, None

    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)

    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    cnt = max(contours, key=cv2.contourArea)

    if cv2.contourArea(cnt) < 0.02 * (w * h):
        return None, None

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None, None

    cx_local = int(M["m10"] / M["m00"])
    cy_local = int(M["m01"] / M["m00"])

    max_dist = -1
    fx_local, fy_local = None, None

    # Prefer points above centroid
    for p in cnt[:, 0, :]:
        px, py = int(p[0]), int(p[1])
        if py > cy_local:
            continue
        dx = px - cx_local
        dy = py - cy_local
        d2 = dx*dx + dy*dy
        if d2 > max_dist:
            max_dist = d2
            fx_local, fy_local = px, py

    # Fallback: farthest overall
    if fx_local is None:
        max_dist = -1
        for p in cnt[:, 0, :]:
            px, py = int(p[0]), int(p[1])
            dx = px - cx_local
            dy = py - cy_local
            d2 = dx*dx + dy*dy
            if d2 > max_dist:
                max_dist = d2
                fx_local, fy_local = px, py

    if fx_local is None:
        return None, None

    fx = fx_local + x
    fy = fy_local + y
    cx = cx_local + x
    cy = cy_local + y

    return (fx, fy), (cx, cy)


# ---------------------------------------------------------
# MAIN TEST LOOP (with vertical side paddle)
# ---------------------------------------------------------

def main():
    cap = cv2.VideoCapture(0)  # change index if needed

    hog = create_hog(
        win_size=HOG_WIN_SIZE,
        block_size=BLOCK_SIZE,
        block_stride=BLOCK_STRIDE,
        cell_size=CELL_SIZE,
        nbins=NBINS
    )
    clf = load(SVM_MODEL_PATH)

    prev_tip = None
    prev_time = None
    ema_tip = None
    alpha = 0.3  # fingertip smoothing

    # Paddle state (center y + which side)
    paddle_cy = None
    paddle_side = None   # "left" or "right"

    frame_idx = 0
    hand_box = None

    t_prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        frame_idx += 1

        H, W = frame.shape[:2]

        # FPS estimation
        t_now = time.time()
        dt_frame = t_now - t_prev
        fps = 1.0 / dt_frame if dt_frame > 0 else 0.0
        t_prev = t_now

        # Draw middle exclusion band (for visualization)
        ex_x1_small = EXCLUDE_X1_FRAC * (W * DETECT_SCALE)
        ex_x2_small = EXCLUDE_X2_FRAC * (W * DETECT_SCALE)
        ex_y1_small = EXCLUDE_Y1_FRAC * (H * DETECT_SCALE)
        ex_y2_small = EXCLUDE_Y2_FRAC * (H * DETECT_SCALE)

        ex_x1 = int(ex_x1_small / DETECT_SCALE)
        ex_x2 = int(ex_x2_small / DETECT_SCALE)
        ex_y1 = int(ex_y1_small / DETECT_SCALE)
        ex_y2 = int(ex_y2_small / DETECT_SCALE)

        cv2.rectangle(display, (ex_x1, ex_y1), (ex_x2, ex_y2),
                      (0, 255, 255), 1)
        cv2.putText(display, "No-detect middle zone",
                    (max(ex_x1, 10), max(ex_y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)

        # Re-detect every N frames, or if we lost it
        if frame_idx % DETECT_EVERY_N == 0 or hand_box is None:
            hand_box, best_score = detect_finger_hog(display, hog, clf)

        fingertip, centroid = None, None
        if hand_box is not None:
            x, y, w, h = hand_box
            cv2.rectangle(display, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)

            fingertip, centroid = find_fingertip_in_box(display, hand_box)

        vx, vy = 0.0, 0.0
        if fingertip is not None:
            fx, fy = fingertip

            # Smooth fingertip
            if ema_tip is None:
                ema_tip = np.array([fx, fy], dtype=np.float32)
            else:
                ema_tip = alpha * np.array([fx, fy], dtype=np.float32) + \
                          (1 - alpha) * ema_tip

            # Draw fingertip + centroid
            cv2.circle(display, (int(ema_tip[0]), int(ema_tip[1])),
                       6, (0, 0, 255), -1)
            if centroid is not None:
                cv2.circle(display, centroid, 4, (255, 0, 0), -1)

            # Velocity (pixels/sec) using unsmoothed tip
            if prev_tip is not None and prev_time is not None:
                dt = t_now - prev_time
                if dt > 0:
                    vx = (fx - prev_tip[0]) / dt
                    vy = (fy - prev_tip[1]) / dt

            prev_tip = (fx, fy)
            prev_time = t_now

            # -------------------------
            # Paddle control logic
            # -------------------------
            tip_x = ema_tip[0]
            tip_y = ema_tip[1]

            # Decide side: if tip is left of middle band -> left, right of band -> right
            if tip_x < ex_x1:
                paddle_side = "left"
            elif tip_x > ex_x2:
                paddle_side = "right"
            # if tip is inside the middle band, keep previous side

            target_cy = tip_y  # vertical control

            if paddle_cy is None:
                paddle_cy = target_cy
            else:
                # Smooth paddle vertical movement
                paddle_cy = PADDLE_ALPHA * target_cy + \
                            (1 - PADDLE_ALPHA) * paddle_cy

        else:
            ema_tip = None
            prev_tip = None
            prev_time = None
            # Keep last paddle_cy/side so it doesn't disappear

        # Draw paddle (vertical, on left or right side)
        if paddle_side is not None and paddle_cy is not None:
            half_h = PADDLE_HEIGHT // 2
            cy_clamped = int(max(half_h, min(H - half_h, paddle_cy)))

            if paddle_side == "left":
                x1 = PADDLE_X_OFFSET
                x2 = x1 + PADDLE_WIDTH
            else:  # "right"
                x2 = W - PADDLE_X_OFFSET
                x1 = x2 - PADDLE_WIDTH

            y1 = cy_clamped - half_h
            y2 = cy_clamped + half_h

            cv2.rectangle(display, (x1, y1), (x2, y2),
                          (255, 0, 0), -1)
            cv2.putText(display, f"Paddle ({paddle_side})",
                        (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 1)

        # HUD text
        cv2.putText(display, f"FPS: {fps:.1f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

        if fingertip is not None:
            cv2.putText(display, f"vy: {vy:.1f}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

        cv2.imshow("Finger + Vertical Paddle (test)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
