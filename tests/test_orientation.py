import numpy as np

def compute_normal(wrist, index, pinky):
    v1 = np.array(index) - np.array(wrist)
    v2 = np.array(pinky) - np.array(wrist)
    normal = np.cross(v1, v2)
    return normal

# Right hand, palm to camera (Selfie view)
# In mirror view, your right hand is on the right.
# Mirror means X=0 is left, X=1 is right.
# Your thumb is leftmost on your hand, but if you hold right hand palm to face, thumb is LEFT.
# Pinky is RIGHT.
# Landmarks: Index(5), Pinky(17)
# Index is landmark 5. Pinky is 17.
wrist = [0.5, 0.5, 0]
index = [0.4, 0.4, 0] # Index left of wrist
pinky = [0.6, 0.4, 0] # Pinky right of wrist
norm = compute_normal(wrist, index, pinky)
print(f"Right hand (Index Left of Pinky): {norm}")

# Left hand, palm to camera (Selfie view)
# Index is landmark 5. Pinky is 17.
# Index would be to the RIGHT of the pinky.
# Index: (0.6, 0.4), Pinky: (0.4, 0.4)
index_l = [0.6, 0.4, 0]
pinky_l = [0.4, 0.4, 0]
norm_l = compute_normal(wrist, index_l, pinky_l)
print(f"Left hand (Index Right of Pinky): {norm_l}")
