"""
So far I have not found a better idea to maintain a single reference on lock objects among all threads and when using
queues objects it becomes very slow (this must be investigated later so that we can remove this part which is not really
cool)

@author: nidragedd
"""
import threading

# Locks that must be acquired to get access to output frames
lock = threading.Lock()
lock_gray = threading.Lock()

# Single and global references to output frame for a live streaming (must be acquired through a lock)
output_frame = None
output_frame_gray = None
