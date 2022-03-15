from math import cos, sin, degrees, atan2, acos, pi


def angleFromCoordinate(lat1, long1, lat2, long2):
    """Compute a bearing (heading) in order to reach point 2 starting from point 1
    See also: http://www.movable-type.co.uk/scripts/latlong.html

    Args:
        lat1 (float): latitude (y-coordinate) of point 1
        long1 (float): longitude (x-coordinate) of point 1
        lat2 (float): latitude (y-coordinate) of point 2
        long2 (float): longitude (x-coordinate) of point 1

    Returns:
        float: Heading to take to reach point 2 starting from point 1
    """
    dLon = (long2 - long1)
    y = sin(dLon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon)
    brng = atan2(y, x)
    brng = degrees(brng)
    brng = (brng + 360) % 360
    return brng


def vr_trigo_bijection(angle):
    """Convert a trigonometric angle to a bearing/heading (and vice versa)

    Args:
        angle (float): the initial angle (either trigonometric or bearing)

    Returns:
        float: the converted angle (either bearing or trigonometric)
    """
    angle_rad = angle * pi / 180
    cos_new_angle = sin(angle_rad)
    sin_new_angle = cos(angle_rad)

    if sin_new_angle > 0:
        return round(degrees(acos(cos_new_angle)), 0)
    else:
        return round(360 - degrees(acos(cos_new_angle)), 0)


def compute_barycenter(points):
    """Compute a ponderate barycenter from a list of points

    Args:
        points (list): list of points (float) [[x,y],[a.b]]

    Returns:
        list: the barycenter as a list of 2 floats [x,y]
    """
    x = 0.
    y = 0.
    sum_coeff = 0
    for i, p in enumerate(points):
        coeff = 1/(i+1)
        sum_coeff += coeff
        x += coeff * p[0]
        y += coeff * p[1]
    x /= sum_coeff
    y /= sum_coeff
    return [x, y]


def filter_wp_angle(lat, lon, waypoints, threshold=5):
    """Filter out waypoints which are too "different" from the first waypoint, in order to avoid over-smoothing

    Args:
        lat (float): latitude of the starting point (the boat) Y-coordinate
        lon (float): longitude of the starting point (the boat) X-coordinate
        waypoints (list): list of the waypoints computed by the isochrones (float) [[x,y],[a,b]]
        threshold (int, optional): The max angle allowed to keep a waypoint. Defaults to 5.

    Returns:
        list: filtered list of waypoints (float)
    """

    # Example 1: keep the 2 WP
    #                                               - boat
    #                        o
    # o
    #
    # Example 2: Not keeping the 2d WP
    #                                               - boat
    #                        o
    #
    #
    #
    #                    o
    filtered_wp = [waypoints[0]]

    lat = lat * pi / 180
    lon = lon * pi / 180

    for i, wp in enumerate(waypoints):
        wp_rad_0 = wp[0] * pi / 180
        wp_rad_1 = wp[1] * pi / 180
        if i == 0:
            heading_first_wp = angleFromCoordinate(
                lat, lon, wp_rad_1, wp_rad_0)
            continue
        heading_current_wp = angleFromCoordinate(
            lat, lon,  wp_rad_1, wp_rad_0)
        if adjust_predicted_angle(heading_current_wp, heading_first_wp, threshold) != heading_current_wp:
            # If this method returns a different angle, it means that the angle is "out of bound"
            break
        filtered_wp.append(wp)
    return filtered_wp


def compute_next_wp(lat, lon, waypoints, collisions):
    """Compute the point to use as a target for the heading prediction

    Args:
        lat (float): latitude of the starting point (the boat) Y-coordinate
        lon (float): longitude of the starting point (the boat) X-coordinate
        waypoints (list): list of the waypoints computed by the isochrones (float) [[x,y],[a,b]]
        collisions (list): collisions list, same length as the waypoints one

    Returns:
        list: target point as a float list [x,y]
    """
    waypoints_no_coll = []

    # Keep only the waypoints that do not have a collision (between the boat and the waypoint)
    for i, c in enumerate(collisions):
        if c == 0:
            waypoints_no_coll.append(waypoints[i])
        else:
            break

    if waypoints_no_coll == []:
        print('The waypoints seems to all have collisions, trying to find one nevertheless')
        # We've got a problem, 2d try: get the first waypoint that does not give a collision
        for i, c in enumerate(collisions):
            if c == 0:
                print(
                    f'Found a waypoint at index {i} without collision: {waypoints[i]}')
                return waypoints[i]
        # If we're still there, we're really in trouble, because all the waypoints yield a collision
        # TODO: what to return in this case? Re-create a new waypoint from scratch?
        print('Did not find a waypoint without collision, returning the first one')
        return waypoints[0]

    waypoints_filtered = filter_wp_angle(lat, lon, waypoints_no_coll)
    if len(waypoints_filtered) == 1:
        # In this case the next WPs are "too different" from the first one, so don't consider them
        print('The next waypoints are too different from the first one, returning the first one only')
        return waypoints[0]

    # Regular case: get barycenter of the first few points that don't give a collisionS
    print('Returning a combination of some waypoints')
    return compute_barycenter(waypoints_filtered)


def adjust_predicted_angle(predicted_angle, target_angle_compass, threshold=5):
    """Apply some thresholding to the predicted bearing/heading to make sure it's somewhat close to the target one

    Args:
        predicted_angle (float): prediction (output of the ML model)
        target_angle_compass (float): the target angle (output of the isochrones)
        threshold (int, optional): Maximum threshold between the target and the prediction. Defaults to 5.

    Returns:
        float: Adjusted bearing/heading
    """
    min_threshold = (target_angle_compass - threshold) % 360
    max_threshold = (target_angle_compass + threshold) % 360

    # "Around 0/360 case"
    #     target
    #  max       min
    if abs(min_threshold - max_threshold) > (2 * threshold):
        min_threshold, max_threshold = max_threshold, min_threshold
        #    target
        #  max          min
        #          prediction
        #        abs(target-180)

        if predicted_angle > min_threshold and predicted_angle <= (target_angle_compass + 180) % 360:
            return min_threshold
        #    target
        #  max          min
        #          prediction
        #        abs(target-180)
        elif predicted_angle < max_threshold and predicted_angle > (target_angle_compass + 180) % 360:
            return max_threshold
    else:
        # "Regular" case
        #        - min
        #             - target
        #        - max
        if predicted_angle < min_threshold:
            #     - predicted
            #        - min
            #             - target
            #        - max
            return min_threshold
        elif predicted_angle > max_threshold:
            #        - min
            #             - target
            #        - max
            #     - predicted
            return max_threshold
    return predicted_angle
