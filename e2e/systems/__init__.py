"""
Systems module for different entity linking implementations
"""
from .base_system import BaseSystem, EntityLink, LinkingResult
from .ranking_system import RankingSystem
from .onenet_system import OneNetSystem
from .graph_system import GraphSystem

__all__ = ['BaseSystem', 'EntityLink', 'LinkingResult', 'RankingSystem', 'OneNetSystem', 'GraphSystem']
