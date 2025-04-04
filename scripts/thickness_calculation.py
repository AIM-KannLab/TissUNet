import numpy as np
from skimage import measure
from skimage.draw import polygon


def get_slice_range(z_index, slice_range=15):
    """
    Get 15 values above a given z_index.
    Offset the z_index by 10 slices which is 1cm to ensure the eye sockets do not skew
    the skull thickness estimation, since these regions are more irregular.

    Parameters:
        z_index (int): The center index to calculate the range around.

    Returns:
        list: A list of z indices to 15 above the given z_index.
    """
    # Generate the range
    return list(
        range(z_index + 10 , z_index + slice_range + 11)
    )  # offset 10 slices (1cm) to exclude eye sockets


def find_longest_contour(mask_slice, level=0.55):
    """
    Identify the longest contour in the binary mask.
    """
    contours = measure.find_contours(mask_slice, level=level)
    if not contours:
        return None
    longest_contour = max(contours, key=len)
    return longest_contour[:, 1], longest_contour[:, 0], contours

def fill_holes_in_contours(mask_slice, level=0.55):
    """
    Process and modify contours based on specific conditions.

    Parameters:
        mask_slice (np.ndarray): The binary mask (2D array) containing 0s and 1s.
        contours (list of np.ndarray): List of contours, each a NumPy array of points.

    Returns:
        np.ndarray: Modified mask_slice after processing contours.
    """
    contours = measure.find_contours(mask_slice, level=level)
    # 1. Check if there are at least 3 contours
    if len(contours) < 3:
        #print("Fewer than 3 contours found. Skipping modification.")
        return mask_slice

    #print("Proceeding with modification. Found at least 3 contours.")

    # Sort contours by length (descending)
    sorted_contours = sorted(contours, key=len, reverse=True)
    longest_contour = sorted_contours[0]
    second_longest_contour = sorted_contours[1]
    remaining_contours = sorted_contours[2:]

    # 2. Verify the longest contour contains a mix of 0s and 1s
    def check_values_in_contour(contour, mask_slice):
        x_coords = contour[:, 1]
        y_coords = contour[:, 0]
        rr, cc = polygon(y_coords, x_coords, mask_slice.shape)
        values = mask_slice[rr, cc]
        return np.all(values == 0), np.all(values == 1), not (np.all(values == 0) or np.all(values == 1))

    is_all_zeros, is_all_ones, is_mixed = check_values_in_contour(longest_contour, mask_slice)
    if not is_mixed:
        #print("Longest contour does not contain a mix of 0s and 1s. Skipping modification.")
        return mask_slice

    # 3. Check remaining contours and flip values in the mask if they contain only 0s
    for i, contour in enumerate(remaining_contours):
        is_all_zeros, is_all_ones, is_mixed = check_values_in_contour(contour, mask_slice)
        if is_all_zeros:
            #print(f"Flipping values inside contour {i + 1} (remaining contours).")
            x_coords = contour[:, 1]
            y_coords = contour[:, 0]
            rr, cc = polygon(y_coords, x_coords, mask_slice.shape)
            mask_slice[rr, cc] = 1  # Flip values from 0 to 1
    return mask_slice

def calculate_contour_perimeter(contour):
    """
    Calculate the perimeter of a contour.
    """
    return np.sum(np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1)))

def compare_longest_contours(contours, threshold_ratio=0.5):
    """
    Compare the lengths of the two longest contours in a binary mask.

    Parameters:
        mask_slice (np.ndarray): contours extracted from the binary mask.
        threshold_ratio (float): Minimum allowable ratio of the second contour's 
                                 length to the longest contour.

    Returns:
        dict: Results containing the lengths of the two longest contours,
              and a flag indicating whether the second contour is too short.
    """
    lengths = sorted((calculate_contour_perimeter(c) for c in contours), reverse=True)
    longest, second_longest = lengths[:2]

    return {
        "lengths": (longest, second_longest),
        "flagged": second_longest < threshold_ratio * longest,
    }


def calculate_checkpoints(x_coords, y_coords, num_checkpoints=100):
    """
    Calculate evenly spaced checkpoints along the contour.
    Vectorized version for improved performance.
    """
    # Calculate cumulative distances along the contour
    distances = np.cumsum(np.sqrt(np.diff(x_coords) ** 2 + np.diff(y_coords) ** 2))
    distances = np.insert(distances, 0, 0)  # Insert initial zero distance
    
    # Create evenly spaced intervals
    total_length = distances[-1]
    equal_intervals = np.linspace(0, total_length, num_checkpoints, endpoint=False)
    
    # Find the closest points on the contour for each interval
    checkpoints = []
    for interval in equal_intervals:
        idx = np.argmin(np.abs(distances - interval))
        checkpoints.append((x_coords[idx], y_coords[idx]))
    
    return checkpoints

def calculate_thickness(
    mask_binary, checkpoints, contour, neighborhood_size=2, initial_offset=0.1
):
    """
    Calculate thickness at each checkpoint based on arrow length.

    Parameters:
        mask_binary (ndarray): Binary mask of the object.
        checkpoints (list): List of (x, y) checkpoint coordinates.
        contour (ndarray): Contour points of the shape.
        neighborhood_size (int): Number of points to consider for tangent calculation.
        initial_offset (float): Offset to avoid boundary issues.

    Returns:
        list: Thickness values at each checkpoint.
    """

    def find_tangent_and_perpendicular(idx, contour, neighborhood_size):
        """
        Find the tangent and perpendicular vector at a contour point index.

        Parameters:
            idx (int): Index of the contour point.
            contour (ndarray): Contour points of the shape.
            neighborhood_size (int): Number of points to consider for tangent calculation.

        Returns:
            tuple: Tangent and perpendicular vectors (both normalized).
        """
        neighborhood_indices = np.arange(
            idx - neighborhood_size, idx + neighborhood_size + 1
        ) % len(contour)
        neighborhood_points = contour[neighborhood_indices]

        dx = np.mean(np.diff(neighborhood_points[:, 1]))
        dy = np.mean(np.diff(neighborhood_points[:, 0]))
        tangent = np.array([dx, dy])
        tangent = tangent / np.linalg.norm(tangent) if np.linalg.norm(tangent) > 0 else np.array([0, 1])

        perp_vector = np.array([-tangent[1], tangent[0]])  # Perpendicular vector
        return tangent, perp_vector
    def extend_in_direction(
        mask_binary, x, y, perp_vector, direction, initial_offset, step_size
    ):
        """
        Extend the arrow in a given direction to calculate its length.
        """
        step = initial_offset  # Start from the initial offset
        direction_length = 0  # Initialize the length

        while True:
            # Calculate the next position
            test_y = y + direction * step * perp_vector[1]
            test_x = x + direction * step * perp_vector[0]
            int_test_y, int_test_x = int(round(test_y)), int(round(test_x))

            # Check if we are outside the mask boundaries or outside the object
            if (
                int_test_y < 0
                or int_test_y >= mask_binary.shape[0]
                or int_test_x < 0
                or int_test_x >= mask_binary.shape[1]
                or mask_binary[int_test_y, int_test_x] != 1
            ):
                break  # Stop if outside boundaries or mask value is not 1

            # Update the direction length and increment step
            direction_length = step - initial_offset
            step += step_size

        return direction_length
    
    # Vectorize the calculation of closest contour points to each checkpoint
    thicknesses = []
    step_size = 0.01
    
    # Pre-compute distances from all contour points to all checkpoints
    contour_points = np.array([(y, x) for x, y in zip(contour[:, 1], contour[:, 0])])  # Shape: (n_contour, 2)
    checkpoint_points = np.array([(y, x) for x, y in checkpoints])  # Shape: (n_checkpoints, 2)
    
    for i, point in enumerate(checkpoints):
        x, y = point
        
        # Find closest contour point to this checkpoint
        dists = np.sqrt(np.sum((contour_points - np.array([y, x])) ** 2, axis=1))
        idx = np.argmin(dists)
        
        # Get tangent and perpendicular vectors
        _, perp_vector = find_tangent_and_perpendicular(idx, contour, neighborhood_size)

        # Calculate thickness in both directions
        total_arrow_length = 0
        for direction in [-1, 1]:
            total_arrow_length += extend_in_direction(
                mask_binary,
                x,
                y,
                perp_vector,
                direction,
                initial_offset,
                step_size,
            )

        thicknesses.append(total_arrow_length)
    
    return thicknesses                


def calculate_statistics(thicknesses):
    """
    Calculate statistical measures for thickness values.
    """
    thicknesses = np.array(thicknesses)
    return {
        "mean": np.mean(thicknesses),
        "median": np.median(thicknesses),
        "0.025 percentile": np.percentile(thicknesses, 2.5),
        "0.975 percentile": np.percentile(thicknesses, 97.5),
        "count": len(thicknesses),
    }

def assign_checkpoints_to_quadrants(checkpoints, thicknesses, center_x, center_y):
    """
    Assign checkpoints and thicknesses to their respective quadrants.
    Vectorized version for improved performance.
    """
    # Convert inputs to numpy arrays
    points = np.array(checkpoints)
    thickness_array = np.array(thicknesses)
    
    # Create masks for each quadrant
    top_left_mask = (points[:, 1] < center_y) & (points[:, 0] < center_x)
    top_right_mask = (points[:, 1] < center_y) & (points[:, 0] >= center_x)
    bottom_left_mask = (points[:, 1] >= center_y) & (points[:, 0] < center_x)
    bottom_right_mask = (points[:, 1] >= center_y) & (points[:, 0] >= center_x)
    
    # Return dictionary with quadrant assignments
    return {
        "top_left": thickness_array[top_left_mask].tolist(),
        "top_right": thickness_array[top_right_mask].tolist(),
        "bottom_left": thickness_array[bottom_left_mask].tolist(),
        "bottom_right": thickness_array[bottom_right_mask].tolist()
    }