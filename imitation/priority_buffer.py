import  heapq
import itertools
from copy import deepcopy
import numpy as np

class TrajObj():
  def __init__(self, ret, sz, uid):
    """
    Create an object from trajectory return (ret) and length (sz)
    """
    self.ret = ret
    self.sz = sz
    self.uid = uid

  def __eq__(self, other):
    return False

  def __gt__(self, other):
    """
    Rules for comparing two trajectories
    """
    assert isinstance(other, type(self))
    if self.ret > other.ret and self.sz > other.sz: return True
    if other.ret > self.ret and other.sz > self.sz: return False
    if self.ret/self.sz > other.ret/other.sz: return True
    if (self.ret == other.ret) and (self.sz == other.sz) and (self.uid > other.uid): return True
    return False

class TrajReplay():
    def __init__(self, capacity):
        self.capacity = capacity
        self.traj_replay = {}
        self.traj_pq = []
        self.counter = itertools.count()
        self.sync_required = True
        self.pointer = 0
        self.obs = self.acs = self.wts = None

    def add_path(self, path):

        uid = next(self.counter)
        path['obs'] = np.array(path['obs'])
        path['acs'] = np.array(path['acs'])

        # Create meta-data for a priority-queue entry
        pqe = TrajObj(path['return'], len(path['obs']), uid)

        # If at capacity, check if the return for this trajectory
        # is greater than the minimum in the replay
        if len(self.traj_replay) == self.capacity:
            min_pqe, _, _ = self.traj_pq[0]
            if pqe > min_pqe:
                _, min_uid, _ = heapq.heappop(self.traj_pq)
                print('Traj with rew:{}, len:{} removed!'.format(min_pqe.ret, min_pqe.sz))
                del self.traj_replay[min_uid]
            else:
                return False

        # Add to replay; deepcopy since we delete the path in main.py
        print('[Traj with rew:{}, len:{} added!'.format(pqe.ret, pqe.sz))
        full_entry = [pqe, uid, deepcopy(path)]
        self.traj_replay[uid] = full_entry
        heapq.heappush(self.traj_pq, full_entry)
        self.sync_required = True
        return True

    def __len__(self):
        return self.obs.shape[0]

    def _calculate_weights(self):
        """
        compute normalized weights from path returns
        """
        wts = np.array([e[2]['return'] for e in self.traj_replay.values()])

        # shift so that all wts are non-negative
        if min(wts) < 0: wts = wts - min(wts)

        # default (if all paths have 0 score)
        if np.sum(wts) == 0:
            wts = np.ones_like(wts)

        lens = [e[2]['obs'].shape[0] for e in self.traj_replay.values()]
        wts_repeated = []
        for i, wt in enumerate(wts.tolist()):
            wts_repeated.extend([wt]*lens[i])
        return np.array(wts_repeated).reshape(-1, 1)

    def sync(self):
        if self.sync_required:
            self.obs = np.concatenate([e[2]['obs'] for e in self.traj_replay.values()], axis=0)
            self.acs = np.concatenate([e[2]['acs'] for e in self.traj_replay.values()], axis=0)
            self.wts = self._calculate_weights()

            self.pointer = 0
            self.sync_required = False
