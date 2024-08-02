import cv2
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from scipy import ndimage
from scipy.signal import find_peaks

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
        # print(combined)
        # print("-------------------------------")
        combined = ndimage.gaussian_filter1d(combined, sigma=2)
        # print(combined)
        gradient = np.diff(combined, axis=0)
        # print(gradient)
        # print("-------------------------------")
        gradient = gradient * 2 * pow(2*math.pi, 0.5)
        # print(gradient)
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


def remove_outliers(points, q1=25, q3=75, iqr_factor=0.5):
    """Remove outliers using the Interquartile Range (IQR) method."""
    q1_x, q3_x = np.percentile(points[:, 0], [q1, q3])
    q1_y, q3_y = np.percentile(points[:, 1], [q1, q3])
    
    iqr_x = q3_x - q1_x
    iqr_y = q3_y - q1_y
    
    lower_bound_x = q1_x - iqr_factor * iqr_x
    upper_bound_x = q3_x + iqr_factor * iqr_x
    lower_bound_y = q1_y - iqr_factor * iqr_y
    upper_bound_y = q3_y + iqr_factor * iqr_y
    
    inliers_mask = (
        (points[:, 0] >= lower_bound_x) & (points[:, 0] <= upper_bound_x) &
        (points[:, 1] >= lower_bound_y) & (points[:, 1] <= upper_bound_y)
    )
    
    inliers = points[inliers_mask]
    outliers = points[~inliers_mask]
    
    return inliers, outliers


def find_first_peak(gradients, peak_threshold=10, valley_type='all'):
    gradients = np.array(gradients).flatten()

    if valley_type == 'negative':
        # 寻找负峰谷（向下的谷）
        peaks, _ = find_peaks(-gradients, height=peak_threshold)
        if len(peaks) > 0:
            return int(peaks[0])
        else:
            # 如果没有找到峰值，返回最小值的索引
            return int(np.argmin(gradients))

    elif valley_type == 'positive':
        # 寻找正峰谷（向上的峰）
        peaks, _ = find_peaks(gradients, height=peak_threshold)
        if len(peaks) > 0:
            return int(peaks[0])
        else:
            # 如果没有找到峰值，返回最大值的索引
            return int(np.argmax(gradients))

    elif valley_type == 'all':
        # 寻找正峰和负谷
        positive_peaks, _ = find_peaks(gradients, height=peak_threshold)
        negative_peaks, _ = find_peaks(-gradients, height=peak_threshold)

        # 如果都没找到，返回绝对值最大的点的索引
        if len(positive_peaks) == 0 and len(negative_peaks) == 0:
            return int(np.argmax(np.abs(gradients)))

        # 如果只找到一种类型的峰，返回第一个
        if len(positive_peaks) == 0:
            return int(negative_peaks[0])
        if len(negative_peaks) == 0:
            return int(positive_peaks[0])

        # 如果两种都找到了，返回最先出现的那个
        return int(min(positive_peaks[0], negative_peaks[0]))

    else:
        raise ValueError("valley_type must be 'positive', 'negative', or 'all'")

def draw_max_gradient_midpoints(image, interpolated_points, gradient_values, base_line, peak_threshold=10, flag='all'):
    """Draw the midpoints of the segments with the maximum gradient."""
    all_max_points = []
    for segment_points, gradients in zip(interpolated_points, gradient_values):
        # if flag == 'all':
        #     max_gradient_index = np.argmax(np.abs(gradients))  # 查找绝对值最大的值 all
        # elif flag == 'positive':
        #     max_gradient_index = np.argmax(gradients)  # 查找最大值 positive 从黑到白
        # elif flag == 'negative':
        #     max_gradient_index = np.argmin(gradients)  # 查找最小值 negative 从白到黑
        max_gradient_index = find_first_peak(gradients, peak_threshold, valley_type=flag)
        print(f"Max index ({flag}): {max_gradient_index}, Value: {gradients[max_gradient_index]}")
        max_point = segment_points[max_gradient_index]
        # print(max_point)
        all_max_points.append(max_point)
        # print(all_max_points)

    all_max_points = np.array(all_max_points)

    # Remove outliers
    inliers, outliers = remove_outliers(all_max_points)

    # Fit a line to the inliers
    line_point, line_direction = fit_line_to_points(inliers)

    base_line_start = np.array(base_line[0])
    base_line_end = np.array(base_line[1])

    projected_start = project_point_on_line(base_line_start, line_point, line_direction)
    projected_end = project_point_on_line(base_line_end, line_point, line_direction)

    inliers = np.round(inliers, decimals=3)
    outliers = np.round(outliers, decimals=3)
    projected_start = np.round(projected_start, decimals=3)
    projected_end = np.round(projected_end, decimals=3)

    return inliers, outliers, np.array([projected_start, projected_end])

def generate_parallel_lines(line, interval, half_length, orientation='P'):
    start_x, start_y = line[0]
    end_x, end_y = line[1]

    # 计算输入线段的方向向量
    dx = end_x - start_x
    dy = end_y - start_y
    length = np.sqrt(dx ** 2 + dy ** 2)

    # 计算单位方向向量
    unit_dx = dx / length
    unit_dy = dy / length

    # 计算垂直单位向量
    perp_dx = -unit_dy
    perp_dy = unit_dx

    # 计算可以生成的平行线段的数量
    num_lines = int(length / interval) + 1

    lines = []
    for i in range(num_lines):
        # 计算中点
        t = i * interval / length
        mid_x = start_x + t * dx
        mid_y = start_y + t * dy

        # 计算垂直线段的起点和终点
        start_perp_x = mid_x - half_length * perp_dx
        start_perp_y = mid_y - half_length * perp_dy
        end_perp_x = mid_x + half_length * perp_dx
        end_perp_y = mid_y + half_length * perp_dy

        if orientation == 'P':
            # 正向：起点在左侧，终点在右侧
            lines.append([start_perp_x, start_perp_y])  # 起点
            lines.append([end_perp_x, end_perp_y])  # 终点
        elif orientation == 'N':
            # 逆向：起点在右侧，终点在左侧
            lines.append([end_perp_x, end_perp_y])  # 起点
            lines.append([start_perp_x, start_perp_y])  # 终点
        else:
            raise ValueError("Invalid orientation value. Use 'P' for positive or 'N' for negative.")

    return np.array(lines)

def process_image_with_lines(img, base_line, flag, interval=1, half_length=10, orientation='P', peak_threshold=10, pixel_interval=3):
    """intrval是直线间隔，half_length是半长，pixel_interval是像素间隔"""
    points = generate_parallel_lines(base_line, interval, half_length, orientation)
    img, points_interpolated, pixel_values = calculate_pixel_values(img, points, pixel_interval)
    grad = calculate_gradient(pixel_values)
    # print(grad)
    # if position == '上' or position == '右':
    #     flag = 'positive'
    # elif position == '下' or position == '左':
    #     flag = 'negative'
    print(f"Processing image with flag: {flag}")
    inliers, outliers, fitted_line = draw_max_gradient_midpoints(img, points_interpolated, grad, base_line, peak_threshold, flag)

    # Round the final return values to three decimal places
    inliers = np.round(inliers, decimals=3)
    outliers = np.round(outliers, decimals=3)
    fitted_line = np.round(fitted_line, decimals=3)
    # 第一个是拟合直线所用的点，第二个是拟合直线的两个端点，第三个是离散点（用不上）
    return inliers, fitted_line, outliers

if __name__ == "__main__":

    # image_path = './images/pzd1.png'
    # base_line = np.array([[100,970], [1700, 970]])
    # position = '上'

    # image_path = 'images/pzd3.png'
    # base_line = np.array([[1050, 100], [1050, 2300]])
    # flag = '右'

    image_path = 'images/pzd4.png'
    base_line = np.array([[991, 100], [983, 2300]])
    # base_line = np.array([[971, 10], [963, 2400]])
    flag = 'negative'

    # image_path = './images/pzd2.png'
    # base_line = np.array([[100,1070], [1700, 1070]])
    # position = '下'

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    interval = 1
    half_length = 20
    orientation = 'P'
    peak_threshold = 10

    inliers, fitted_line, outliers = process_image_with_lines(img, base_line, flag, interval, half_length, orientation, peak_threshold)
    print("Inliers:", inliers)
    # print("Outliers:", outliers)
    print("Fitted Line Endpoints:", fitted_line)

    # Create a color version of the image for visualization
    viz_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw the base line
    # cv2.line(viz_img, tuple(base_line[0]), tuple(base_line[1]), (0, 255, 0), 1)

    # Draw the inliers in green
    # for point in inliers:
    #     cv2.circle(viz_img, tuple(point.astype(int)), 3, (0, 255, 0), -1)

    # Draw the outliers in red
    # for point in outliers:
    #     cv2.circle(viz_img, tuple(point.astype(int)), 3, (0, 0, 255), -1)

    # Draw the fitted line in blue
    cv2.line(viz_img, tuple(fitted_line[0].astype(int)), tuple(fitted_line[1].astype(int)), (255, 0, 0), 1)

    # Create a window and display the image
    cv2.namedWindow('Image with Fitted Line', cv2.WINDOW_NORMAL)
    cv2.imshow('Image with Fitted Line', viz_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()