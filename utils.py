import numpy as np
import math


def time_elapse_parser(time_elapsed):
    # time_elapsed = time.time() - start_time
    m_elapsed, s_elapsed = divmod(time_elapsed, 60)
    h_elapsed, m_elapsed = divmod(m_elapsed, 60)
    return '%d:%02d:%02d' % (h_elapsed, m_elapsed, s_elapsed)

