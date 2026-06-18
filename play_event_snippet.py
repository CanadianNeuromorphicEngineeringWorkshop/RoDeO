"""
Visualise a 35 ms event snippet (.npy with fields x, y, p, t).

Left  : "video" of the events played back in 1 ms steps (latest window only).
Right : all events of the whole snippet accumulated into one image.

ON events = green, OFF events = red. Background = black.

Usage:
    python view_event_snippet.py path/to/snippet_t00035ms.npy
    (or set SNIPPET below and run with no args)
"""

import sys
import numpy as np
import cv2

# ---------------------------------------------------------------------------
SNIPPET   = "/Users/sarkaliskova/Documents/phd/codes/CNEW/experimental/converted_events/mug/mug_full/mug_5_90_-15_zoom_events_t00000ms.npy"
WIDTH, HEIGHT = 346, 260
STEP_MS   = 1            # playback step
PLAY_MS   = 40           # wall-clock ms per frame (playback speed)
LOOP      = True         # loop the left-side animation
# ---------------------------------------------------------------------------


def colour_frame(x, y, p):
    """Render events into a BGR image: ON=green, OFF=red."""
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    on  = p == 1
    off = ~on
    img[y[off], x[off]] = (0, 0, 255)    # red  (OFF)
    img[y[on],  x[on]]  = (0, 255, 0)    # green (ON)
    return img


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (255, 255, 255), -1)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else SNIPPET
    ev = np.load(path)
    if ev.shape[0] == 0:
        print(f"Empty snippet: {path}")
        return

    x = ev["x"].astype(np.int32)
    y = ev["y"].astype(np.int32)
    p = ev["p"].astype(np.int8)
    t = ev["t"].astype(np.int64)

    # clip coords defensively
    np.clip(x, 0, WIDTH - 1, out=x)
    np.clip(y, 0, HEIGHT - 1, out=y)

    t0 = t.min()
    rel_ms = (t - t0) / 1000.0
    span_ms = rel_ms.max()
    print(f"{path}\n  {ev.shape[0]} events, span {span_ms:.2f} ms")

    # right panel: ALL events of the snippet, fixed
    right = label(colour_frame(x, y, p), f"All events ({ev.shape[0]})")

    n_steps = int(np.ceil(span_ms / STEP_MS)) + 1
    gap = np.full((HEIGHT, 10, 3), 255, np.uint8)

    while True:
        for s in range(n_steps):
            lo = s * STEP_MS
            hi = lo + STEP_MS
            m = (rel_ms >= lo) & (rel_ms < hi)      # events in this 1 ms slice
            left = label(colour_frame(x[m], y[m], p[m]),
                         f"t = {lo:>3d}-{hi:>3d} ms   ({int(m.sum())} ev)")

            grid = np.hstack([left, gap, right])
            cv2.imshow("event snippet  (left: 1ms video | right: all events)", grid)
            if cv2.waitKey(PLAY_MS) & 0xFF in (27, ord("q")):   # Esc or q
                cv2.destroyAllWindows()
                return

        if not LOOP:
            break

    print("Press any key in the window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()