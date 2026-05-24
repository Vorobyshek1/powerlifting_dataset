from powerlift_analyzer.geometry import angle_deg, angle_to_vertical_deg


def test_right_angle():
    assert round(angle_deg((1, 0), (0, 0), (0, 1))) == 90


def test_straight_angle():
    assert round(angle_deg((-1, 0), (0, 0), (1, 0))) == 180


def test_vertical_angle():
    assert round(angle_to_vertical_deg((0, 1), (0, 0))) == 0
