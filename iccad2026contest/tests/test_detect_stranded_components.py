import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from detect_stranded_components import detect_stranded, find_components  # noqa: E402


def test_single_tightly_packed_group_is_one_component():
    positions = [(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 2.0, 2.0), (0.0, 2.0, 2.0, 2.0)]
    result = detect_stranded(positions)
    assert result["component_count"] == 1
    assert not result["stranded"]


def test_two_components_close_together_are_not_flagged():
    # Small, ordinary-looking spacing (much less than the largest block's
    # own dimension) shouldn't trigger stranding.
    positions = [(0.0, 0.0, 4.0, 4.0), (4.5, 0.0, 4.0, 4.0)]
    result = detect_stranded(positions)
    assert result["component_count"] == 2
    assert not result["stranded"]


def test_far_isolated_block_is_flagged_stranded():
    # Mirrors the real block-50 pattern: one block sitting far from the
    # dominant mass, with a gap much larger than any single block's size.
    main_mass = [(x * 2.0, 0.0, 2.0, 2.0) for x in range(10)]
    isolated = [(200.0, 200.0, 2.0, 2.0)]
    result = detect_stranded(main_mass + isolated)
    assert result["component_count"] == 2
    assert result["stranded"]
    assert result["components"][0]["blocks"] == [10]


def test_find_components_merges_touching_rectangles():
    positions = [(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 2.0, 2.0), (100.0, 100.0, 2.0, 2.0)]
    components = find_components(positions)
    assert len(components) == 2
    sizes = sorted(len(c) for c in components)
    assert sizes == [1, 2]
