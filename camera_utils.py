import cv2


def set_window_pos(name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(name, 0, 0)


def safe_close(cap):
    cap.release()
    cv2.destroyAllWindows()


def init_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # 640
    cap.set(4, 480)  # 480
    time.sleep(2.)
    _, frame = cap.read()
    rows, cols, _ = frame.shape
    center = int(cols / 2)
    return cap, center, rows


if __name__ == "__main__":
	pass