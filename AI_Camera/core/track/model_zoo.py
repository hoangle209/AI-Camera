import numpy as np
from third_parties.track.byte_track.byte_tracker import STrack

def do_track_byte_track(
        byte_tracker, 
        dets=None,
        xyxy=None,
        confidence=None,
        class_id=None,
    ):
    """
    img_info and img_size are specified if needed only, 
    in update function the two arguments are used to define box scale
    """
    if dets is None:
        dets = np.concatenate([
                        xyxy, 
                        confidence[..., None],
                        class_id[..., None]], axis=1)
    tracked_stracks = byte_tracker.update(dets)

    if len(tracked_stracks) == 0:
        return np.array([])

    xyxy     = np.array([STrack.tlwh_to_tlbr(strack.tlwh) for strack in tracked_stracks]).astype("int")
    score    = np.array([strack.score for strack in tracked_stracks])[:, None]
    cls_id   = np.array([strack.class_id for strack in tracked_stracks]).astype("int")[:, None]
    track_id = np.array([strack.track_id for strack in tracked_stracks])[:, None]

    tracklets = np.concatenate([xyxy, score, cls_id, track_id], axis=1)
    return tracklets


# Tracker zoo
tracker_zoo = {
    "byte_track": do_track_byte_track
}