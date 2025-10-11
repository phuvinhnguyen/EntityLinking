"""
Systems module for different entity linking implementations
"""
from .base_system import BaseSystem, EntityLink, LinkingResult
from .ranking_system import RankingSystem
from .onenet_system import OneNetSystem
from .simple_system import SimpleSystem

__all__ = ['BaseSystem', 'EntityLink', 'LinkingResult', 'RankingSystem', 'OneNetSystem', 'SimpleSystem']
