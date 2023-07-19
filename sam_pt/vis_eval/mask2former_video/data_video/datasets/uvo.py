UVO_CATEGORIES_V1_CLASS_AGNOSTIC = [
    {"color": [106, 0, 228], "isthing": 1, "id": 1, "name": "object"},
]


def _get_uvo_v1_instances_meta():
    thing_ids = [k["id"] for k in UVO_CATEGORIES_V1_CLASS_AGNOSTIC if k["isthing"] == 1]
    assert len(thing_ids) == 1, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in UVO_CATEGORIES_V1_CLASS_AGNOSTIC if k["isthing"] == 1]
    thing_colors = [k["color"] for k in UVO_CATEGORIES_V1_CLASS_AGNOSTIC if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret
