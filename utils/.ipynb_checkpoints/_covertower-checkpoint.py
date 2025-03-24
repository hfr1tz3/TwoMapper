import numpy as np

def _remove_duplicate_covers(cover_masks):
    # Remove any cover which is a duplicate to another
    unique_cover_masks = [cover_masks[0]]
    for i, cover in enumerate(cover_masks):
        if i > 0:
            prev_size = cover_masks[i-1].shape
            size = cover.shape
            if prev_size != size:
                unique_cover_masks.append(cover)
            else:
                if not np.all(np.equal(cover, cover_masks[i-1])):
                    unique_cover_masks.append(cover)
    return unique_cover_masks