import numpy as np
import cv2
import glob

NPZ_FILE = "/Users/sarkaliskova/Documents/phd/codes/CNEW/experimental/data/ROI_events/bottle_0_00005_full.npz"   
W, H     = 346, 260                                     # input resolution (DAVIS346 resolution)

# ── load ──────────────────────────────────────────────────────────────────────
data     = np.load(NPZ_FILE, allow_pickle=True) 
ev       = data["events"].item() if data["events"].ndim == 0 else data["events"] #events inlcludeing x,y, timestamp and polarity
cx       = int(data["cx"].item()) #x coordinate of ROI center
cy       = int(data["cy"].item()) #y coordinate of ROI center
roi_size = int(data["roi_size"].item()) 
roi_half = roi_size // 2

print(f"Events: {len(ev)}  |  cx={cx} cy={cy}  |  roi_size={roi_size}")
print(f"Event fields: {ev.dtype.names}")

xs  = ev["x"].astype(np.int32) #x coordinates of events
ys  = ev["y"].astype(np.int32) #y coordinates of events
pol = ev["polarity"].astype(bool) 
t = ev["timestamp"].astype(np.int64)
#──────────────────────────────────────────────────────────────────────


# ── visualisation
full_bgr = np.zeros((H, W, 3), dtype=np.uint8)
m = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)

full_bgr[ys[m &  pol], xs[m &  pol]] = (255, 255, 255)   # positive: white
full_bgr[ys[m & ~pol], xs[m & ~pol]] = (120, 120, 120)   # negative: grey

# ROI box + centroid
x1 = max(0, cx - roi_half);  x2 = min(W - 1, cx + roi_half)
y1 = max(0, cy - roi_half);  y2 = min(H - 1, cy + roi_half)
cv2.rectangle(full_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
cv2.drawMarker(full_bgr, (cx, cy), (0, 255, 0),
               cv2.MARKER_CROSS, markerSize=14, thickness=1)

roi_bgr = np.zeros((roi_size, roi_size, 3), dtype=np.uint8)

# keep only events inside the ROI bounding box
in_roi = (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
rx = (xs[in_roi] - x1).astype(np.int32)   # local coords
ry = (ys[in_roi] - y1).astype(np.int32)
rp = pol[in_roi]

rh, rw = roi_bgr.shape[:2]
valid  = (rx >= 0) & (rx < rw) & (ry >= 0) & (ry < rh)
roi_bgr[ry[valid &  rp], rx[valid &  rp]] = (255, 255, 255)
roi_bgr[ry[valid & ~rp], rx[valid & ~rp]] = (120, 120, 120)

# visualise ROI bounding box
cv2.rectangle(roi_bgr, (0, 0), (rw - 1, rh - 1), (0, 255, 255), 1)
roi_display = cv2.resize(roi_bgr, (int(roi_size * H / roi_size), H),
                         interpolation=cv2.INTER_NEAREST)

# add labels
def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (180, 180, 0), 1, cv2.LINE_AA)
    return out

full_bgr    = label(full_bgr,    f"Full frame  cx={cx} cy={cy}  roi={roi_size}px")
roi_display = label(roi_display, f"ROI crop  {x2-x1}x{y2-y1}px")

# pad roi_display to same height if needed
if roi_display.shape[0] != H:
    pad = H - roi_display.shape[0]
    roi_display = cv2.copyMakeBorder(roi_display, 0, pad, 0, 0,
                                     cv2.BORDER_CONSTANT, value=0)

combined = np.hstack([full_bgr, roi_display])

cv2.namedWindow("Events + ROI", cv2.WINDOW_NORMAL)
cv2.imshow("Events + ROI", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()