import cv2
import numpy as np

def process_lines(image, points, pixel_interval=1, threshold=20):
    def calculate_pixel_values(img, points, pixel_interval):
        height, width = img.shape

        if np.any((points[:, 0] < 0) | (points[:, 0] >= width) | (points[:, 1] < 0) | (points[:, 1] >= height)):
            raise ValueError("点位超出图片范围")

        total_distance = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        num_points = np.ceil(total_distance / pixel_interval).astype(int) + 1

        t = np.linspace(0, 1, np.max(num_points))
        x = np.interp(t, np.arange(len(points)), points[:, 0])
        y = np.interp(t, np.arange(len(points)), points[:, 1])

        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)

        wx, wy = x - x0, y - y0

        pixel_values = (
            (1 - wx) * (1 - wy) * img[y0, x0] +
            wx * (1 - wy) * img[y0, x1] +
            (1 - wx) * wy * img[y1, x0] +
            wx * wy * img[y1, x1]
        )

        pixel_values = pixel_values.reshape(-1, 1).astype(int)

        return np.vstack((x, y)).T, pixel_values

    def calculate_gradient(pixel_values):
        combined = np.concatenate(([pixel_values[0]], pixel_values), axis=0)
        gradient_values = np.diff(combined, axis=0)
        gradient_values = np.round(gradient_values).astype(int)
        return gradient_values

    def find_gradient_midpoints(points, gradient_values, threshold):
        midpoints = []

        for i in range(len(points) - 1):
            if abs(gradient_values[i]) > threshold:
                midpoint = ((points[i][0] + points[i + 1][0]) / 2, (points[i][1] + points[i + 1][1]) / 2)
                midpoints.append(midpoint)

        return midpoints

    # Validate input image
    if image is None or len(image.shape) != 2:
        raise ValueError("Invalid image. Please provide a grayscale image.")

    all_midpoints = []

    for i in range(len(points) - 1):
        # Create line segment from consecutive points
        line_segment = np.array([points[i], points[i + 1]])

        # Calculate pixel values and interpolated points
        points_interpolated, pixel_values = calculate_pixel_values(image, line_segment, pixel_interval)

        # Calculate gradient
        gradient_values = calculate_gradient(pixel_values)

        # Find midpoints
        midpoints = find_gradient_midpoints(points_interpolated, gradient_values, threshold)
        all_midpoints.extend(midpoints)

    return np.array(all_midpoints)

# Example usage:
if __name__ == "__main__":
    image_path = './images/16.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Define points array
    points = np.array([
        [1500, 1480],
        [2000, 1480]
    ])

    midpoints = process_lines(image, points)
    print(midpoints)
