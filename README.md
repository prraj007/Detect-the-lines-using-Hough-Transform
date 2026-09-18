#  Lane Detection

##  Aim

To implement a basic lane detection pipeline using OpenCV by completing missing code segments at specified locations.

---

## Learning Objective

* Understand each stage of image processing
* Learn how to build a complete computer vision pipeline
* Practice writing code in guided sections

**Important Instruction:**
👉 Write code **ONLY in places marked as `# Your Code Here`**
👉 Do NOT modify any other part of the code

---

##  Software Used

* Anaconda – Python 3.7
* Jupyter Notebook / VS Code
* OpenCV (cv2)
* NumPy
* Matplotlib

---

##  Algorithm & Explanation

---

###  Step 1: Import Libraries

```
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

---

###  Step 2: Read the Image

```
img = cv2.imread('lan_img1.jpg')
```

---

###  Step 3: Convert to Grayscale

```
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

---

###  Step 4: Display Images

```
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale')
plt.axis('off')

plt.show()
```

---

###  Step 5: Thresholding

```
threshold = 
cv2.threshold(
    gray, 127, 255, cv2.THRESH_BINARY
)[1]

plt.figure(figsize=(20, 10))
plt.subplot(1, 1, 1)
plt.imshow(threshold, cmap='gray')
plt.title('Threshold')
plt.axis('off')
plt.show()

```

---

###  Step 6: Region of Interest (ROI)

```
roi_vertices = np.array([[[100, 540],
                          [900, 540],
                          [515, 320],
                          [450, 320]]])

mask = np.zeros_like(threshold)

if len(threshold.shape) > 2:
    channel_count = threshold.shape[2]
    ignore_mask_color = (255,) * channel_count
else:
    ignore_mask_color = 255

cv2.fillPoly(mask, roi_vertices, ignore_mask_color)

roi = cv2.bitwise_and(threshold, mask)

plt.figure(figsize=(20, 10))

plt.subplot(1, 3, 1)
plt.imshow(threshold, cmap='gray')
plt.title('Initial threshold')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('Polyfill mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(roi, cmap='gray')
plt.title('Isolated roi')
plt.axis('off')

plt.show()
```

---

### Step 7: Edge Detection (Canny)

```
edges = cv2.Canny(roi, 50, 150)
```

---

###  Step 8: Gaussian Blur

```
canny_blur = cv2.GaussianBlur(edges, (5, 5), 0)

plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.imshow(edges, cmap='gray')
plt.title('Edge detection')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(canny_blur, cmap='gray')
plt.title('Blurred edges')
plt.axis('off')

plt.show()

```

---

###  Step 9: Hough Transform

```
lines = cv2.HoughLinesP(
    canny_blur,
    rho=1,
    theta=np.pi / 180,
    threshold=50,
    minLineLength=50,
    maxLineGap=100
)


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Utility for drawing lines."""
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


hough = np.zeros(
    (img.shape[0], img.shape[1], 3),
    dtype=np.uint8
)

draw_lines(hough, lines)

if lines is not None:
    print(
        "Found {} lines, including: {}".format(
            len(lines), lines[0]
        )
    )
else:
    print("No lines found.")

plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(hough, cv2.COLOR_BGR2RGB))
plt.title('Hough Lines')
plt.axis('off')
plt.show()
```

---

### Step 10: Lane Detection Logic

```
def separate_left_right_lines(lines):
    """Separate left and right lines depending on the slope."""
    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:

                if y1 > y2:
                    # Negative slope = left lane.
                    left_lines.append([x1, y1, x2, y2])

                elif y1 < y2:
                    # Positive slope = right lane.
                    right_lines.append([x1, y1, x2, y2])

    return left_lines, right_lines



def cal_avg(values):
    """Calculate average value."""
    if not (type(values) == 'NoneType'):
        if len(values) > 0:
            n = len(values)
        else:
            n = 1

        return sum(values) / n



def extrapolate_lines(lines, upper_border, lower_border):
    """Extrapolate lines keeping in mind the lower and upper border intersections."""

    slopes = []
    consts = []

    if lines is not None:
        for x1, y1, x2, y2 in lines:

            slope = (y1 - y2) / (x1 - x2)
            slopes.append(slope)

            c = y1 - slope * x1
            consts.append(c)

    avg_slope = cal_avg(slopes)
    avg_consts = cal_avg(consts)

    # Calculate average intersection at lower_border.
    x_lane_lower_point = int(
        (lower_border - avg_consts) / avg_slope
    )

    # Calculate average intersection at upper_border.
    x_lane_upper_point = int(
        (upper_border - avg_consts) / avg_slope
    )

    return [
        x_lane_lower_point,
        lower_border,
        x_lane_upper_point,
        upper_border
    ]



roi_upper_border = 340
roi_lower_border = 540

lanes_img = np.zeros(
    (img.shape[0], img.shape[1], 3),
    dtype=np.uint8
)

lines_left, lines_right = separate_left_right_lines(lines)

lane_left = extrapolate_lines(
    lines_left,
    roi_upper_border,
    roi_lower_border
)

lane_right = extrapolate_lines(
    lines_right,
    roi_upper_border,
    roi_lower_border
)

draw_lines(
    lanes_img,
    [[lane_left]],
    thickness=10
)

draw_lines(
    lanes_img,
    [[lane_right]],
    thickness=10
)


fig = plt.figure(figsize=(20, 20))

ax = fig.add_subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(hough, cv2.COLOR_BGR2RGB))
ax.set_title('Before extrapolation')
ax.axis('off')

ax = fig.add_subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(lanes_img, cv2.COLOR_BGR2RGB))
ax.set_title('After extrapolation')
ax.axis('off')

plt.show()
```

---

##  Expected Output

* Original image
* <img width="474" height="255" alt="image" src="https://github.com/user-attachments/assets/c5aa4c9c-580f-4293-8b06-6a83d7b8ac0f" />

* Grayscale image
* <img width="413" height="252" alt="image" src="https://github.com/user-attachments/assets/fa4acadd-26bf-4ee5-9541-92cf440f0163" />

* Thresholded image
* <img width="898" height="526" alt="image" src="https://github.com/user-attachments/assets/a0d83bb5-ac0c-4062-a34f-6fb78d52eee0" />

* ROI masked image
* <img width="898" height="173" alt="image" src="https://github.com/user-attachments/assets/32cb8119-ed09-43f3-abf6-17c2b0e6b5c9" />

* Edge detected image
* <img width="417" height="252" alt="image" src="https://github.com/user-attachments/assets/771ac60a-eaee-4547-959a-d810f3e73532" />
* Smoothed image
* <img width="418" height="251" alt="image" src="https://github.com/user-attachments/assets/b773dc9b-fc2c-46b5-99df-32ad6ee45abc" />
* Detected lines
* <img width="898" height="529" alt="image" src="https://github.com/user-attachments/assets/c193c962-8221-4c0a-a55e-fb7b62f8cbd2" />

* Final lane detection output
<img width="899" height="250" alt="image" src="https://github.com/user-attachments/assets/79fe155a-3fe3-400e-a15d-a67d1ddca9b2" />

---

##  Instructions

* Fill ONLY in `# Your Code Here` sections
* Do NOT change existing code
* Run step-by-step
* Verify outputs

---

## Result

Thus, the lane detection pipeline is successfully implemented by completing the missing code sections. The system detects and highlights lane lines effectively.

---

##  Developed By

* **Name:** B.PRAVEEN RAJ
* **Register No:** 212225040315
