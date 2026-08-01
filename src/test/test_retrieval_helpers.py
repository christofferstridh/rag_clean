from main.run import (
    consolidate_groupings,
    get_filtered_matches,
    get_min_max_ids,
    group_entries,
    is_unique_to_window,
)


def test_is_unique_to_window_allows_non_overlapping_matches() -> None:
    existing_matches = [(1, 1, "a", "file-a"), (2, 10, "a", "file-a")]
    current_match = (3, 5, "a", "file-a")

    assert is_unique_to_window(existing_matches, current_match, group_window_size=3) is True


def test_is_unique_to_window_rejects_overlapping_matches() -> None:
    existing_matches = [(1, 1, "a", "file-a"), (2, 10, "a", "file-a")]
    current_match = (3, 2, "a", "file-a")

    assert is_unique_to_window(existing_matches, current_match, group_window_size=3) is False


def test_group_entries_groups_same_file_and_window() -> None:
    entry_ids = [10, 11, 20, 21]
    file_names = ["a.txt", "a.txt", "b.txt", "b.txt"]

    assert group_entries(entry_ids, file_names, 1, 2) == [1, 0]


def test_consolidate_groupings_merges_overlapping_groups() -> None:
    grouped_entries = [[0], [0, 1], [2]]

    assert consolidate_groupings(grouped_entries) == [[0, 1], [2]]


def test_get_filtered_matches_keeps_only_unique_matches() -> None:
    search_results = [
        (1, 1, "x", "file-a"),
        (2, 2, "x", "file-a"),
        (3, 3, "x", "file-b"),
    ]

    assert get_filtered_matches(search_results) == [
        (1, 1, "x", "file-a"),
        (3, 3, "x", "file-b"),
    ]


def test_get_min_max_ids_returns_bounds_for_each_group() -> None:
    entry_ids = [10, 11, 20, 21]
    file_names = ["a", "a", "b", "b"]
    combined_groups = [[0, 1], [2, 3]]

    min_ids, max_ids = get_min_max_ids(entry_ids, file_names, combined_groups, 2)

    assert min_ids == [8, 18]
    assert max_ids == [13, 23]
