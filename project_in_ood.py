import cv2
import numpy as np

class MOSSE:
    def __init__(self, frame, rect, learning_rate=0.125, num_warps=8):
        """
        frame: first frame as a grayscale float32 image
        rect: tuple (x, y, w, h) of initial ROI in pixels
        """
        x, y, w, h = rect
        self.learning_rate = learning_rate
        self.num_warps = num_warps
        self.pos = (x + w/2, y + h/2)       # center position
        self.size = (w, h)                  # (width, height)
        self.psr_threshold = 8.0

        # Create a Hanning (cosine) window for preprocessing
        self.window = np.outer(np.hanning(int(h)), np.hanning(int(w))).astype(np.float32)

        # Desired correlation output: a 2D Gaussian peaked at center
        self.G = self._create_gaussian_response((int(w), int(h)), sigma=2.0)

        # Initialize accumulators in the frequency domain
        self.Ai = np.zeros_like(self.G, dtype=np.complex64)
        self.Bi = np.zeros_like(self.G, dtype=np.complex64)

        # Build initial filter using synthetic warp samples
        self._init_filter(frame)

    def _preprocess(self, img):
        """Preprocessing: log, normalize, and window"""
        img = np.log(img + 1.0)
        img = (img - img.mean()) / (img.std() + 1e-5)
        return img * self.window

    def _create_gaussian_response(self, size, sigma):
        """Create desired output (Gaussian peak), and FFT it"""
        w, h = size
        x = np.arange(w) - w/2
        y = np.arange(h) - h/2
        xv, yv = np.meshgrid(x, y)
        gauss = np.exp(-0.5 * (xv**2 + yv**2) / sigma**2)
        return np.fft.fft2(gauss.astype(np.float32))

    def _random_warp(self, img):
        """Apply a small random warp to generate synthetic training samples"""
        h, w = img.shape
        # Random rotation, scale, translation
        ang = np.random.uniform(-0.1, 0.1)  # radians
        sc = np.random.uniform(0.9, 1.1)
        c, s = np.cos(ang)*sc, np.sin(ang)*sc
        M = np.array([[ c, -s, 0],
                      [ s,  c, 0]], dtype=np.float32)
        # random shifts
        M[0,2] = np.random.uniform(-0.1*w, 0.1*w)
        M[1,2] = np.random.uniform(-0.1*h, 0.1*h)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _init_filter(self, frame):
        """Initialize Ai and Bi by generating warped samples from the first ROI"""
        x, y = int(self.pos[0] - self.size[0]/2), int(self.pos[1] - self.size[1]/2)
        x, y = max(0,x), max(0,y)
        patch = frame[y:y+int(self.size[1]), x:x+int(self.size[0])]
        patch = cv2.resize(patch, (int(self.size[0]), int(self.size[1])))

        for _ in range(self.num_warps):
            warped = self._random_warp(patch)
            X = self._preprocess(warped)
            F = np.fft.fft2(X)
            self.Ai += self.G * np.conj(F)
            self.Bi += F * np.conj(F)

    def _psr(self, resp):
        """Compute Peak-to-Sidelobe Ratio"""
        resp_real = resp.real
        h, w = resp_real.shape
        # find peak
        cy, cx = np.unravel_index(np.argmax(resp_real), resp_real.shape)
        peak = resp_real[cy, cx]
        # mask out a small window around the peak
        sidelobe = resp_real.copy()
        r = 5  # exclusion radius
        y1, y2 = max(0, cy-r), min(h, cy+r+1)
        x1, x2 = max(0, cx-r), min(w, cx+r+1)
        sidelobe[y1:y2, x1:x2] = 0
        mean = sidelobe.mean()
        std = sidelobe.std()
        return (peak - mean) / (std + 1e-5)

    def update(self, frame):
        """
        Track object in new frame.
        Returns: (x, y, w, h) of updated bounding box, PSR value
        """
        w, h = int(self.size[0]), int(self.size[1])
        # 1) extract patch at current position (with border-reflect)
        patch = cv2.getRectSubPix(frame, (w, h), self.pos)
        X = self._preprocess(patch)
        F = np.fft.fft2(X)

        # 2) form filter and correlate
        H = self.Ai / (self.Bi + 1e-5)
        resp = np.fft.ifft2(H * F)
        resp_real = resp.real

        # 3) find peak & displacement
        cy, cx = np.unravel_index(np.argmax(resp_real), resp_real.shape)
        dy, dx = cy - h//2, cx - w//2
        new_pos = (self.pos[0] + dx, self.pos[1] + dy)

        # 4) compute PSR
        psr = self._psr(resp)
        if psr > self.psr_threshold:
            # commit to new position
            self.pos = new_pos

            # 5) re-extract patch at updated pos for learning
            patch2 = cv2.getRectSubPix(frame, (w, h), self.pos)
            X2 = self._preprocess(patch2)
            F2 = np.fft.fft2(X2)

            # 6) exponential moving average update
            self.Ai = (1 - self.learning_rate) * self.Ai + \
                      self.learning_rate * (self.G * np.conj(F2))
            self.Bi = (1 - self.learning_rate) * self.Bi + \
                      self.learning_rate * (F2 * np.conj(F2))

        # return integer bbox and PSR
        x_new = int(self.pos[0] - w/2)
        y_new = int(self.pos[1] - h/2)
        return (x_new, y_new, w, h), psr


if __name__ == "__main__":
    import sys, time    
    import matplotlib.pyplot as plt

    video_path = "Media/ballMoving.mov"
    x, y, w, h = 7,373,269,280
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to open video")
        sys.exit(1)

    # Prepare first frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tracker = MOSSE(gray, (x, y, w, h))

    # Lists for performance data
    fps_list = []
    psr_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Measure start time
        start_time = time.perf_counter()

        (x1, y1, w1, h1), psr = tracker.update(gray)

        # Measure end time
        end_time = time.perf_counter()

        # Calculate elapsed time
        elapsed = end_time - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_list.append(fps)
        psr_list.append(psr)


        # Draw bounding box and PSR
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"PSR: {psr:.2f}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("MOSSE Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


        # Generate plots
    plt.figure()
    plt.plot(fps_list)
    plt.title("Real-Time Performance (FPS over Frames)")
    plt.xlabel("Frame Index")
    plt.ylabel("FPS")
    plt.show()

    plt.figure()
    plt.plot(psr_list)
    plt.title("Tracking Confidence (PSR over Frames)")
    plt.xlabel("Frame Index")
    plt.ylabel("PSR")
    plt.show()