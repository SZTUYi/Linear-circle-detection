import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

def calculate_pixel_values(img, points, pixel_interval):
    height, width = img.shape

    pixel_values = []
    interpolated_points = []

    for i in range(0, len(points) - 1, 2):
        line_points = points[i:i + 2]

        if np.any((line_points[:, 0] < 0) | (line_points[:, 0] >= width) | (line_points[:, 1] < 0) | (
                line_points[:, 1] >= height)):
            raise ValueError("点位超出图片范围")

        total_distance = np.sqrt(np.sum(np.diff(line_points, axis=0) ** 2, axis=1))
        num_points = np.ceil(total_distance / pixel_interval).astype(int) + 1

        t = np.linspace(0, 1, np.max(num_points))
        x = np.interp(t, np.arange(len(line_points)), line_points[:, 0])
        y = np.interp(t, np.arange(len(line_points)), line_points[:, 1])

        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)

        wx, wy = x - x0, y - y0

        values = (
                (1 - wx) * (1 - wy) * img[y0, x0] +
                wx * (1 - wy) * img[y0, x1] +
                (1 - wx) * wy * img[y1, x0] +
                wx * wy * img[y1, x1]
        )

        pixel_values.append(values.reshape(-1, 1).astype(int))
        interpolated_points.append(np.vstack((x, y)).T)

    return img, interpolated_points, pixel_values


def calculate_gradient(pixel_values):
    gradient_values = []
    for values in pixel_values:
        combined = np.concatenate(([values[0]], values), axis=0)
        gradient = np.diff(combined, axis=0)
        gradient = np.round(gradient).astype(int)
        gradient_values.append(gradient)
    return gradient_values


def project_point_on_line(point, line_point, line_direction):
    """Project a point onto a line defined by a point and a direction vector."""
    point_vector = point - line_point
    projection_length = np.dot(point_vector, line_direction) / np.dot(line_direction, line_direction)
    projection_point = line_point + projection_length * line_direction
    return projection_point


def fit_line_to_points(points):
    """Fit a line to the given points and return the direction vector and a point on the line."""
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_

    line_point = np.array([0, intercept])
    line_direction = np.array([1, slope])
    return line_point, line_direction


def draw_max_gradient_midpoints(image, interpolated_points, gradient_values, base_line):
    all_max_points = []

    # Calculate the maximum gradient point for each segment
    for segment_points, gradients in zip(interpolated_points, gradient_values):
        # Find the index of the maximum absolute gradient value
        max_index = np.argmax(np.abs(gradients))
        max_point = segment_points[max_index]
        all_max_points.append(max_point)

    all_max_points = np.array(all_max_points)

    # Fit a line to the maximum gradient points
    line_point, line_direction = fit_line_to_points(all_max_points)

    # Find projections of the base line endpoints onto the fitted line
    base_line_start = np.array(base_line[0])
    base_line_end = np.array(base_line[1])

    projected_start = project_point_on_line(base_line_start, line_point, line_direction)
    projected_end = project_point_on_line(base_line_end, line_point, line_direction)

    all_max_points = all_max_points.astype(int)
    projected_start = projected_start.astype(int)
    projected_end = projected_end.astype(int)

    return all_max_points, np.array([projected_start, projected_end])


def generate_parallel_lines(line, interval, half_length):
    start_x, start_y = line[0]
    end_x, end_y = line[1]

    # Calculate direction vector of the input line
    dx = end_x - start_x
    dy = end_y - start_y
    length = np.sqrt(dx ** 2 + dy ** 2)

    # Calculate unit direction vector
    unit_dx = dx / length
    unit_dy = dy / length

    # Calculate perpendicular unit vector
    perp_dx = -unit_dy
    perp_dy = unit_dx

    # Calculate number of lines
    num_lines = int(length / interval) + 1

    lines = []
    for i in range(num_lines):
        # Calculate midpoint
        t = i * interval / length
        mid_x = start_x + t * dx
        mid_y = start_y + t * dy

        # Calculate start and end points of perpendicular line
        start_perp_x = mid_x - half_length * perp_dx
        start_perp_y = mid_y - half_length * perp_dy
        end_perp_x = mid_x + half_length * perp_dx
        end_perp_y = mid_y + half_length * perp_dy

        lines.append([start_perp_x, start_perp_y])  # start point
        lines.append([end_perp_x, end_perp_y])  # end point

    return np.array(lines)


def process_image_with_lines(img, base_line, interval=1, half_length=10, pixel_interval=1):
    points = generate_parallel_lines(base_line, interval, half_length)
    img, points_interpolated, pixel_values = calculate_pixel_values(img, points, pixel_interval)
    grad = calculate_gradient(pixel_values)
    max_gradient_points, fitted_line = draw_max_gradient_midpoints(img, points_interpolated,
                                                                  grad, base_line)

    return max_gradient_points, fitted_line


# Example usage:
if __name__ == "__main__":
    # image_path = './images/21.png'
    # base_line = np.array([[991, 100], [983, 2300]])

    image_path = './images/21.png'
    base_line = np.array([[991, 100], [983, 2300]])
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    interval = 1
    half_length = 10

    max_gradient_points, fitted_line = process_image_with_lines(img, base_line, interval, half_length)
    print("Max Gradient Points:", max_gradient_points)
    print("Fitted Line Endpoints:", fitted_line)

    # Create a color version of the image for visualization
    viz_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw the base line
    # cv2.line(viz_img, tuple(base_line[0]), tuple(base_line[1]), (255, 0, 0), 2)

    # Draw the max gradient points
    # for point in max_gradient_points:
    #     cv2.circle(viz_img, tuple(point), 3, (0, 0, 255), -1)

    # Draw the fitted line
    cv2.line(viz_img, tuple(fitted_line[0]), tuple(fitted_line[1]), (0,255, 0), 2)

    # Create a window and display the image
    cv2.namedWindow('Image with Fitted Line', cv2.WINDOW_NORMAL)
    cv2.imshow('Image with Fitted Line', viz_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()