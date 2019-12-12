"""
So far I have not found a better idea to maintain a single reference on lock objects among all threads and when using
queues objects it becomes very slow (this must be investigated later so that we can remove this part which is not really
cool)

@author: nidragedd
"""
import threading

######################################
#  LIVE MOTION DETECTION SECTION
######################################
# Locks that must be acquired to get access to output frames
motion_detection_lock = threading.Lock()
motion_detection_lock_gray = threading.Lock()
# Single and global references to output frame for a live motion (must be acquired through a lock)
motion_detection_output_frame = None
motion_detection_output_frame_gray = None

######################################
#  LIVE OBJECT DETECTION SECTION
######################################
object_detection_lock = threading.Lock()
object_detection_output_frame = None
