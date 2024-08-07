import csv
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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
        # interpolated_points.append(np.vstack((x, y)).T)
        # 反转插值点的顺序
        interpolated_points.append(np.vstack((x, y)).T[::-1])

    return img, interpolated_points, pixel_values

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
    # print(len(inliers))
    outliers = points[~inliers_mask]
    # print(len(outliers))

    return inliers, outliers


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

    # print(np.array(lines))
    return np.array(lines)


def find_first_peak(gradients, peak_threshold=10, valley_type='both'):
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

    elif valley_type == 'both':
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
        raise ValueError("valley_type must be 'positive', 'negative', or 'both'")

def calculate_gradient(pixel_values):
    reversed_pixel_values = [values[::-1] for values in pixel_values]
    # print(reversed_pixel_values)
    return [np.diff(np.concatenate(([values[0]], values), axis=0), axis=0).astype(int) for values in reversed_pixel_values]

def draw_max_gradient_midpoints(image, interpolated_points, gradient_values, base_line, flag='all'):
    """Draw the midpoints of the segments with the maximum gradient."""
    all_max_points = []
    # print(gradient_values)
    for segment_points, gradients in zip(interpolated_points, gradient_values):
        if flag == 'all':
            max_index = find_first_peak(gradients)
            # print(f"Max index (all): {max_index}, Value: {gradients[max_index]}")
        elif flag == 'positive':
            max_index = find_first_peak(gradients, valley_type='positive')
            # print(f"Max index (positive): {max_index}, Value: {gradients[max_index]}")
        elif flag == 'negative':
            max_index = find_first_peak(gradients, valley_type='negative')
            # print(f"Max index (negative): {max_index}, Value: {gradients[max_index]}")

        max_point = segment_points[max_index]
        # print(f"Max point: {max_point}")
        all_max_points.append(max_point)

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


def process_image_with_lines(img, base_line, position, interval=1, half_length=1200, pixel_interval=2):
    """intrval是直线间隔，half_length是半长，pixel_interval是像素间隔"""
    points = generate_parallel_lines(base_line, interval, half_length)
    img, points_interpolated, pixel_values = calculate_pixel_values(img, points, pixel_interval)
    grad = calculate_gradient(pixel_values)
    # print(grad)
    if position == 'first_positive':
        flag = 'positive'
    elif position == 'first_negative':
        flag = 'negative'
    elif position == 'all':
        flag = 'all'
    inliers, outliers, fitted_line = draw_max_gradient_midpoints(img, points_interpolated, grad, base_line, flag)

    # Round the final return values to three decimal places
    inliers = np.round(inliers, decimals=3)
    outliers = np.round(outliers, decimals=3)
    fitted_line = np.round(fitted_line, decimals=3)
    # 第一个是拟合直线所用的点，第二个是拟合直线的两个端点，第三个是离散点（用不上）
    return inliers, fitted_line, outliers

def calculate_midpoint(line):
    return (line[0] + line[1]) / 2

def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def process_image(image_path, base_line1, base_line2, position1, position2, output_folder):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 处理第一条线
    inliers1, fitted_line1, outliers1 = process_image_with_lines(img, base_line1, position1)

    # 处理第二条线
    inliers2, fitted_line2, outliers2 = process_image_with_lines(img, base_line2, position2)

    # 计算拟合直线的中点
    midpoint1 = calculate_midpoint(fitted_line1)
    midpoint2 = calculate_midpoint(fitted_line2)

    # 计算中点之间的距离
    distance = calculate_distance(midpoint1, midpoint2)

    # 创建彩色图像用于可视化
    viz_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 绘制基准线和拟合线
    cv2.line(viz_img, tuple(base_line1[0].astype(int)), tuple(base_line1[1].astype(int)), (0, 255, 0), 1)
    cv2.line(viz_img, tuple(base_line2[0].astype(int)), tuple(base_line2[1].astype(int)), (0, 255, 0), 1)
    cv2.line(viz_img, tuple(fitted_line1[0].astype(int)), tuple(fitted_line1[1].astype(int)), (0, 0, 255), 1)
    cv2.line(viz_img, tuple(fitted_line2[0].astype(int)), tuple(fitted_line2[1].astype(int)), (0, 0, 255), 1)

    # 绘制中点
    cv2.circle(viz_img, tuple(midpoint1.astype(int)), 5, (255, 0, 0), -1)
    cv2.circle(viz_img, tuple(midpoint2.astype(int)), 5, (255, 0, 0), -1)

    # 绘制中点之间的连线
    cv2.line(viz_img, tuple(midpoint1.astype(int)), tuple(midpoint2.astype(int)), (255, 255, 0), 1)

    # 保存处理后的图片
    output_image_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_image_path, viz_img)

    return fitted_line1, fitted_line2, distance


def process_folder(folder_path, base_line1, base_line2, position1, position2, output_folder):
    results = []

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            fitted_line1, fitted_line2, distance = process_image(image_path, base_line1, base_line2, position1,
                                                                 position2, output_folder)
            # print(fitted_line1)

            results.append({
                'filename': filename,
                'fitted_line1': fitted_line1.tolist(),
                'fitted_line2': fitted_line2.tolist(),
                'midpoint1' : calculate_midpoint(fitted_line1),
                'midpoint2' : calculate_midpoint(fitted_line2),
                'distance': distance
            })

            print(f"Processed {filename}: Distance = {distance:.3f}")

    return results


def save_results_to_csv(results, output_file):
    distance_diff_list = []
    midpoint1_y_diff_list = []

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'fitted_line1_start_x', 'fitted_line1_start_y',
                      'fitted_line1_end_x', 'fitted_line1_end_y',
                      'fitted_line2_start_x', 'fitted_line2_start_y',
                      'fitted_line2_end_x', 'fitted_line2_end_y', 'midpoint1_y', 'midpoint2_y', 'distance',
                      'midpoint1_y_diff', 'midpoint2_y_diff', 'distance_diff']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        if results:
            first_midpoint1_y = results[0]['midpoint1'][1]
            first_midpoint2_y = results[0]['midpoint2'][1]
            first_distance = results[0]['distance']

            for result in results:
                midpoint1_y_diff = result['midpoint1'][1] - first_midpoint1_y
                midpoint2_y_diff = result['midpoint2'][1] - first_midpoint2_y
                distance_diff = result['distance'] - first_distance

                distance_diff_list.append(distance_diff)
                midpoint1_y_diff_list.append(midpoint1_y_diff)

                writer.writerow({
                    'filename': result['filename'],
                    'fitted_line1_start_x': result['fitted_line1'][0][0],
                    'fitted_line1_start_y': result['fitted_line1'][0][1],
                    'fitted_line1_end_x': result['fitted_line1'][1][0],
                    'fitted_line1_end_y': result['fitted_line1'][1][1],
                    'fitted_line2_start_x': result['fitted_line2'][0][0],
                    'fitted_line2_start_y': result['fitted_line2'][0][1],
                    'fitted_line2_end_x': result['fitted_line2'][1][0],
                    'fitted_line2_end_y': result['fitted_line2'][1][1],
                    'midpoint1_y': result['midpoint1'][1],
                    'midpoint2_y': result['midpoint2'][1],
                    'distance': result['distance'],
                    'midpoint1_y_diff': midpoint1_y_diff,
                    'midpoint2_y_diff': midpoint2_y_diff,
                    'distance_diff': distance_diff
                })
    return distance_diff_list, midpoint1_y_diff_list


def linear_regression_analysis(distance_diff, midpoint1_y_diff, output_folder):
    y = np.array(distance_diff)
    x = np.array(midpoint1_y_diff).reshape(-1, 1)

    # 创建并拟合线性回归模型
    model = LinearRegression()
    model.fit(x, y)

    # 打印结果
    print(f"斜率 (a): {model.coef_[0]:.6f}")
    print(f"截距 (b): {model.intercept_:.6f}")

    # 计算 R-squared 值
    r_squared = model.score(x, y)
    print(f"R-squared: {r_squared:.6f}")

    # 绘制原始数据点和拟合线
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='数据点')
    plt.plot(x, model.predict(x), color='red', label='拟合线')
    plt.xlabel('Distance Difference')
    plt.ylabel('Midpoint1 Y Difference')
    plt.title('线性回归拟合')
    plt.legend()
    # 保存图表
    output_path = os.path.join(output_folder, 'linear_regression_fit.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Regression plot saved to {output_path}")

    plt.show()

    return model.coef_[0], model.intercept_, r_squared


if __name__ == "__main__":
    # folder_path = './MindVision_back/'  # 替换为您的图片文件夹路径
    # output_folder = './processed_images/back'  # 处理后的图片保存路径
    # output_file = './processed_images/back/results_back.csv'  # 结果将保存在这个CSV文件中
    folder_path = './MindVision_ahead/'  # 替换为您的图片文件夹路径
    output_folder = './processed_images/ahead'  # 处理后的图片保存路径
    output_file = './processed_images/ahead/results_ahead.csv'  # 结果将保存在这个CSV文件中
    # folder_path = './test/'  # 替换为您的图片文件夹路径
    # output_folder = './processed_images/test'  # 处理后的图片保存路径
    # output_file = './processed_images/test/results_test.csv'  # 结果将保存在这个CSV文件中

    base_line1 = np.array([[100, 2180], [4400, 2180]])
    base_line2 = np.array([[100, 2220], [4400, 2220]])
    position1 = 'first_negative'
    position2 = 'first_positive'

    results = process_folder(folder_path, base_line1, base_line2, position1, position2, output_folder)
    distance_diff, midpoint1_y_diff = save_results_to_csv(results, output_file)

    print(f"Results saved to {output_file}")
    print(f"Processed images saved to {output_folder}")

    # 执行线性回归分析
    slope, intercept, r_squared = linear_regression_analysis(distance_diff, midpoint1_y_diff, output_folder)

    # 将回归结果添加到CSV文件
    with open(output_file, 'a', newline='') as csvfile:
        csvfile.write(f"\nRegression Results\n")
        csvfile.write(f"Slope,{slope:.6f}\n")
        csvfile.write(f"Intercept,{intercept:.6f}\n")
        csvfile.write(f"R-squared,{r_squared:.6f}\n")

    print("Regression results appended to CSV file.")