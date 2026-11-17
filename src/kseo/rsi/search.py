"""小预算 UCT 候选树。无随机 rollout；每个 reward 来自独立实测/明确模拟。"""

import math

from ..util import KSEOError, finite, positive_int


class Tree:
    def __init__(self, *, branching=2, max_depth=3, exploration=1.4):
        self.branching = positive_int(branching, 'branching')
        self.max_depth = positive_int(max_depth, 'max_depth')
        self.exploration = finite(exploration, 'exploration')
        self.nodes = [{'id': 0, 'parent': None, 'depth': 0, 'valid': True, 'closed': False,
                       'visits': 0, 'reward_sum': 0., 'speedup': 1., 'status': 'BASELINE'}]

    def children(self, node):
        return [row for row in self.nodes if row['parent'] == node['id']]

    def expandable(self, node):
        if not node['valid'] or node['closed'] or node['depth'] >= self.max_depth:
            return False
        children = self.children(node)
        return len(children) < self.branching or any(self.expandable(row) for row in children)

    def select(self):
        node = self.nodes[0]
        if not self.expandable(node):
            return None
        while len(self.children(node)) >= self.branching:
            children = [row for row in self.children(node) if self.expandable(row)]
            node = max(children, key=lambda row: (
                row['reward_sum'] / max(1, row['visits']) + self.exploration *
                math.sqrt(math.log(max(2, node['visits'])) / max(1, row['visits'])), -row['id']))
        return node

    def add(self, parent, *, status, speedup=None, evidence=None):
        if parent not in self.nodes or not self.expandable(parent):
            raise KSEOError('候选树父节点不可扩展')
        valid = status == 'LOCAL_VALID'
        speed = finite(speedup, 'speedup') if valid else None
        if valid and speed <= 0:
            raise KSEOError('局部 speedup 必须大于 0')
        reward = max(-1., 1. - 1. / speed) if valid else -1.
        node = {'id': len(self.nodes), 'parent': parent['id'], 'depth': parent['depth'] + 1,
                'status': status, 'valid': valid, 'closed': not valid, 'speedup': speed,
                'visits': 0, 'reward_sum': 0., 'evidence': evidence or {}}
        self.nodes.append(node)
        current = node
        while current is not None:
            current['visits'] += 1
            current['reward_sum'] += reward
            current = self.nodes[current['parent']] if current['parent'] is not None else None
        return node
