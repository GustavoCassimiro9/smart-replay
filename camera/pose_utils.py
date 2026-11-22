# Keypoint indices — COCO format used by YOLOv8 pose
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10

CONF_THRESHOLD = 0.4  # minimum keypoint confidence to trust a coordinate
MIN_SHOULDER_WIDTH_PX = 10  # below this the person is sideways and left/right is unreliable


def has_both_hands_raised(keypoints) -> bool:
    """
    Determine whether a single person has both hands raised.

    Args:
        keypoints: numpy array of shape (17, 3) — columns are (x, y, confidence).

    Returns:
        True if both wrists are above (lower y value than) their respective
        shoulders and all four keypoints are detected with enough confidence.
    """
    if len(keypoints) <= max(LEFT_WRIST, RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER):
        return False

    lw_x, lw_y, lw_c = keypoints[LEFT_WRIST]
    rw_x, rw_y, rw_c = keypoints[RIGHT_WRIST]
    ls_x, ls_y, ls_c = keypoints[LEFT_SHOULDER]
    rs_x, rs_y, rs_c = keypoints[RIGHT_SHOULDER]

    # Reject if any relevant keypoint is below the confidence threshold
    if min(lw_c, rw_c, ls_c, rs_c) < CONF_THRESHOLD:
        return False

    # Image y-axis points downward: a raised hand has a smaller y than the shoulder
    left_raised = lw_y < ls_y
    right_raised = rw_y < rs_y

    return left_raised and right_raised


def has_arms_crossed_x(keypoints) -> bool:
    """
    Determine whether a single person is making an "X" with the forearms.

    The X is detected when:
      1. The wrists are crossed: each wrist is on the opposite side of the
         body from its own shoulder (left wrist past the right wrist).
      2. The elbows are NOT crossed (keep the normal shoulder order). Together
         with (1) this means the forearms intersect, forming the X.
      3. Both wrists are above their elbows (forearms pointing up), which
         covers an X in front of the chest or above the head and rejects
         arms simply folded at the waist.

    Comparing against the shoulder order (instead of a fixed "x is smaller")
    makes it work whether the player faces the camera or has their back to it.

    Args:
        keypoints: numpy array of shape (17, 3) — columns are (x, y, confidence).
    """
    needed = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST)
    if len(keypoints) <= max(needed):
        return False

    ls_x, ls_y, ls_c = keypoints[LEFT_SHOULDER]
    rs_x, rs_y, rs_c = keypoints[RIGHT_SHOULDER]
    le_x, le_y, le_c = keypoints[LEFT_ELBOW]
    re_x, re_y, re_c = keypoints[RIGHT_ELBOW]
    lw_x, lw_y, lw_c = keypoints[LEFT_WRIST]
    rw_x, rw_y, rw_c = keypoints[RIGHT_WRIST]

    if min(ls_c, rs_c, le_c, re_c, lw_c, rw_c) < CONF_THRESHOLD:
        return False

    # Side-on players: left/right order of the shoulders can't be trusted
    shoulder_dx = ls_x - rs_x
    if abs(shoulder_dx) < MIN_SHOULDER_WIDTH_PX:
        return False
    body_side = 1 if shoulder_dx > 0 else -1  # +1 when facing the camera

    wrists_crossed = (lw_x - rw_x) * body_side < 0
    elbows_normal = (le_x - re_x) * body_side > 0
    forearms_up = lw_y < le_y and rw_y < re_y

    return wrists_crossed and elbows_normal and forearms_up
