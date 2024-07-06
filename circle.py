import cv2
import numpy as np

def calculate_pixel_values(img, points, pixel_interval):
    height, width = img.shape
    points = np.clip(points.reshape(-1, 2, 2), [0, 0], [width - 1, height - 1])
    
    pixel_values = []
    interpolated_points = []

    for start, end in points:
        distance = np.linalg.norm(end - start)
        num_points = int(np.ceil(distance / pixel_interval)) + 1
        t = np.linspace(0, 1, num_points)[:, np.newaxis]

        xy = start[np.newaxis, :] * (1 - t) + end[np.newaxis, :] * t
        xy = np.clip(xy, [0, 0], [width - 1, height - 1])

        xy0 = np.floor(xy).astype(int)
        xy1 = np.minimum(xy0 + 1, [width - 1, height - 1])
        w = xy - xy0

        p00 = img[xy0[:, 1], xy0[:, 0]]
        p01 = img[xy0[:, 1], xy1[:, 0]]
        p10 = img[xy1[:, 1], xy0[:, 0]]
        p11 = img[xy1[:, 1], xy1[:, 0]]

        values = (p00 * (1 - w[:, 0]) * (1 - w[:, 1]) +
                  p01 * w[:, 0] * (1 - w[:, 1]) +
                  p10 * (1 - w[:, 0]) * w[:, 1] +
                  p11 * w[:, 0] * w[:, 1])

        pixel_values.append(values.reshape(-1, 1).astype(int))
        interpolated_points.append(xy)

    return img, interpolated_points, pixel_values

def calculate_gradient(pixel_values):
    return [np.diff(np.concatenate(([values[0]], values), axis=0), axis=0).astype(int) for values in pixel_values]

def draw_gradient_threshold_midpoints(image, interpolated_points, gradient_values, flag='all'):
    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    all_midpoints = []

    for segment_points, gradients in zip(interpolated_points, gradient_values):
        if flag == 'all':
            max_gradient_index = np.argmax(np.abs(gradients)) # 查找绝对值最大的值 all
        elif flag == 'positive':
            max_gradient_index = np.argmax(gradients)   # 查找最大值 positive 从黑到白
        elif flag == 'negative':
            max_gradient_index = np.argmin(gradients)   # 查找最小值 negative 从白到黑
        if max_gradient_index < len(segment_points) - 1:
            midpoint = tuple(map(int, (segment_points[max_gradient_index] + segment_points[max_gradient_index + 1]) / 2))
        else:
            midpoint = tuple(map(int, segment_points[-1]))
        cv2.circle(color_img, midpoint, radius=1, color=(0, 0, 255), thickness=-1)
        all_midpoints.append(midpoint)

    return color_img, all_midpoints

def generate_radial_lines(center, radius, angle_interval, line_length):
    angles = np.deg2rad(np.arange(0, 360, angle_interval))
    cx, cy = center
    boundary_points = np.column_stack((
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles)
    ))
    
    start_points = boundary_points - line_length * np.column_stack((np.cos(angles), np.sin(angles)))
    end_points = boundary_points + line_length * np.column_stack((np.cos(angles), np.sin(angles)))
    
    return np.column_stack((start_points, end_points)).reshape(-1, 2, 2)

def fit_circle(points):
    points = np.array(points)
    x, y = points[:, 0], points[:, 1]
    x_m, y_m = np.mean(x), np.mean(y)
    u, v = x - x_m, y - y_m

    Suv, Suu, Svv = np.sum(u*v), np.sum(u**2), np.sum(v**2)
    Suuv, Suvv, Suuu, Svvv = np.sum(u**2 * v), np.sum(u * v**2), np.sum(u**3), np.sum(v**3)

    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuv + Suvv, Svvv + Suuu]) / 2
    uc, vc = np.linalg.solve(A, B)

    xc, yc = uc + x_m, vc + y_m
    R = np.sqrt(uc**2 + vc**2 + (Suu + Svv) / len(points))

    return (xc, yc), R

def remove_outliers(points, factor=0.27):
    """factor: 四分位距倍数，值越小，去除越多离群点，采用四分位距去除离群点"""
    points = np.array(points)
    q1 = np.percentile(points, 25, axis=0)
    q3 = np.percentile(points, 75, axis=0)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    mask = np.all((points >= lower_bound) & (points <= upper_bound), axis=1)
    return points[mask], points[~mask]

def process_image(img, initial_center, initial_radius, angle_interval=1, line_length=20, pixel_interval=1):
    """angel_interval: 角度间隔，line_length: 线段半长，pixel_interval: 像素间隔"""
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.circle(color_img, tuple(map(int, initial_center)), int(initial_radius), (0, 0, 255), 2)

    points = generate_radial_lines(initial_center, initial_radius, angle_interval, line_length)
    img, points_interpolated, pixel_values = calculate_pixel_values(img, points, pixel_interval)
    grad = calculate_gradient(pixel_values)
    _, midpoints = draw_gradient_threshold_midpoints(img, points_interpolated, grad)
    
    midpoints, removed_points = remove_outliers(midpoints)
    fitted_center, fitted_radius = fit_circle(midpoints)

    points = generate_radial_lines(fitted_center, fitted_radius, angle_interval, line_length-10)
    img, points_interpolated, pixel_values = calculate_pixel_values(img, points, pixel_interval)
    grad = calculate_gradient(pixel_values)
    _, midpoints = draw_gradient_threshold_midpoints(img, points_interpolated, grad)
    
    midpoints, removed_points = remove_outliers(midpoints)
    final_fitted_center, final_fitted_radius = fit_circle(midpoints)

    final_fitted_center = tuple(map(lambda x: round(x, 3), final_fitted_center))
    final_fitted_radius = round(final_fitted_radius, 3)

    cv2.circle(color_img, tuple(map(int, final_fitted_center)), int(final_fitted_radius), (0, 255, 0), 2)

    cv2.namedWindow("Circle Fitting Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Circle Fitting Result", color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return final_fitted_center, final_fitted_radius

# if __name__ == "__main__":
#     image_path = './images/23.png'
#     initial_center = (1200, 980)
#     initial_radius = 100
#     angle_interval = 1
#     line_length = 40
#     pixel_interval = 1

#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise ValueError(f"Unable to read image at {image_path}")

#     final_center, final_radius = process_image(img, initial_center, initial_radius)
#     print(f"Fitted circle center: {final_center}")
#     print(f"Fitted circle radius: {final_radius}")

if __name__ == "__main__":

    # initial_center = (2785, 2750)
    # initial_radius = 880
    # initial_center = (2792, 2761)
    # initial_radius = 830
    # 23
    image_path = './images/23.png'
    initial_center = (1200, 980) 
    initial_radius = 100
    # #22
    # image_path = './images/22.png'
    # initial_center = (1180, 1022)
    # initial_radius = 100
    # #20
    image_path = './images/20.png'
    # initial_center = (903, 817)
    initial_center = (883, 817)
    initial_radius = 100
    angle_interval = 1
    line_length = 40
    pixel_interval = 1

    # 20
    # image_path = './images/20.png'
    # initial_center = (1220, 1026)
    # initial_radius = 100
    #
    # angle_interval = 1
    # line_length = 30
    # pixel_interval = 1

    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image at {image_path}")

    # Create a color image for drawing
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 初始圆 (red)
    cv2.circle(color_img, (int(initial_center[0]), int(initial_center[1])), int(initial_radius), (0, 0, 255), 2)

    # 圆上的线 (yellow)
    points = generate_radial_lines(initial_center, initial_radius, angle_interval, line_length)
    points = points.reshape(-1, points.shape[-1])
    # for i in range(0, len(points), 2):
    #     start_point = (int(points[i][0]), int(points[i][1]))
    #     end_point = (int(points[i + 1][0]), int(points[i + 1][1]))
    #     cv2.line(color_img, start_point, end_point, (0, 255, 255), 1)

    # First iteration
    img, points_interpolated, pixel_values = calculate_pixel_values(img, points, pixel_interval)
    grad = calculate_gradient(pixel_values)
    _, midpoints = draw_gradient_threshold_midpoints(img, points_interpolated, grad)
    print(len(midpoints))
    midpoints, removed_points = remove_outliers(midpoints)
    print(len(midpoints))

    # # 保留的点（参与拟合）t (Green)
    # for point in midpoints:
    #     cv2.circle(color_img, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)  # Green
    #
    # # 去除的点 (Red)
    # for point in removed_points:
    #     cv2.circle(color_img, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)  # Red

    fitted_center, fitted_radius = fit_circle(midpoints)
    
    print(f"Fitted circle center1_____________________________________________: {fitted_center}")
    print(f"Fitted circle radius1_____________________________________________: {fitted_radius}")

    # 第一次拟合的圆 (blue)
    # cv2.circle(color_img, (int(fitted_center[0]), int(fitted_center[1])), int(fitted_radius), (255, 0, 0), 2)

    # Second iteration using the fitted circle from the first iteration
    points = generate_radial_lines(fitted_center, fitted_radius, angle_interval, line_length-10)
    img, points_interpolated, pixel_values = calculate_pixel_values(img, points, pixel_interval)
    grad = calculate_gradient(pixel_values)
    _, midpoints = draw_gradient_threshold_midpoints(img, points_interpolated, grad)
    print(len(midpoints))
    midpoints, removed_points = remove_outliers(midpoints)
    print(len(midpoints))

    # # 保留的点（参与拟合） (Orange)
    # for point in midpoints:
    #     cv2.circle(color_img, (int(point[0]), int(point[1])), 2, (0, 165, 255), -1)  # Orange
    #
    # # 去除的点 (Red)
    # for point in removed_points:
    #     cv2.circle(color_img, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)  # Red

    final_fitted_center, final_fitted_radius = fit_circle(midpoints)

    # Round the results to one decimal place
    final_fitted_center = (round(final_fitted_center[0], 3), round(final_fitted_center[1], 3))
    final_fitted_radius = round(final_fitted_radius, 3)

    # 二次拟合的圆(green)
    cv2.circle(color_img, (int(final_fitted_center[0]), int(final_fitted_center[1])), int(final_fitted_radius),(0, 255, 0), 2)

    # Display the result
    cv2.namedWindow("Circle Fitting Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Circle Fitting Result", color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Fitted circle center2_____________________________________________: {final_fitted_center}")
    print(f"Fitted circle radius2_____________________________________________: {final_fitted_radius}")