from collections import defaultdict
from typing import List, Dict, Optional, Iterable, Generic, TypeVar
import time


T = TypeVar('T')
class TrieNode(Generic[T]):
    __slots__ = ('value', 'children')

    def __init__(self, value: Optional[T] = None):
        self.value = value
        self.children: Dict[str, TrieNode[T]] = defaultdict(lambda: TrieNode())


class Trie(Generic[T]):
    def __init__(self):
        self.root: TrieNode[T] = TrieNode()

    def add(self, key: str, value: T):
        node = self.root
        for c in key:
            node = node.children[c]
        node.value = value
    
    def get(self, key: str) -> Optional[T]:
        node: Optional[TrieNode] = self.root
        for c in key:
            node = node.children.get(c)
            if not node:
                return None
        return node.value
    
    def get_wildcard(self, key: str, wildcard_char: str = '?') -> Iterable[T]:
        nodes: Iterable[TrieNode[T]] = [self.root]
        for c in key:
            if c == wildcard_char:
                nodes = tuple(node for next_node in nodes for node in next_node.children.values())
            else:
                nodes = tuple(node for node in (node.children.get(c) for node in nodes) if node is not None)
        return (node.value for node in nodes if node.value is not None)


def main():
    trie: Trie[str] = Trie()
    start_time = time.time()
    print('Loading words...')
    with open('russian-words/russian.txt', 'r', encoding='cp1251') as f:
        words = [w for w in (w.strip().lower() for w in f.readlines()) if len(w) > 0]
    print(f'{len(words)} words loaded in {(time.time() - start_time) * 1000} ms.')

    start_time = time.time()
    print('Adding words...')
    for w in words:
        trie.add(w, w)
    print(f'Added in {(time.time() - start_time) * 1000} ms.')

    test_key = 'баг??'
    print(f'trie.get(\'{test_key}\') =', list(trie.get_wildcard(test_key)))


if __name__ == '__main__':
    main()
