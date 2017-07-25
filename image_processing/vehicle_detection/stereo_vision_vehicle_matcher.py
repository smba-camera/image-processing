from image_processing import util

def match_vehicles_stereo(vehicles_img_0, vehicles_img_1):
    # don't modify original lists
    v_0 = list(vehicles_img_0)
    v_1 = list(vehicles_img_1)

    matching_vehicles = []
    for v in v_0:
        matching_v = find_matching_vehicle(v,v_1)
        if matching_v:
            v_1.remove(matching_v)
        matching_vehicles.append((v, matching_v))
    if len(v_1) > 0:
        for v in v_1:
            matching_vehicles.append((None, v))
    return matching_vehicles

def find_matching_vehicle(v, vehicles):
    mean_p = mean_point(v)
    alpha = 40
    possible_partners = []
    for v2 in vehicles:
        mean_p_v2 = mean_point(v2)
        dist = util.distance(mean_p, mean_p_v2)
        if dist <= alpha:
            # could be a matching partner
            possible_partners.append((v2, dist))

    if len(possible_partners) == 0:
        return None
    if len(possible_partners) == 1:
        return possible_partners[0][0]

    possible_partners.sort(key=lambda x: x[1])
    return possible_partners[0][0]

def mean_point(p):
    mean_p = (
        p[0][0] + 0.5 * (p[1][0] - p[0][0]),
        p[0][1] + 0.5 * (p[1][1] - p[0][1])
    )
    return mean_p

