from __future__ import annotations
from typing import Dict

def sign_index(lon: float) -> int:
    return int(lon // 30)  # 0..11

def d9_navamsa_sign(lon: float) -> int:
    # Each sign is split into 9 parts of 3°20' = 3.333.. deg
    # Rule: movable signs start from same sign, fixed start from 9th sign, dual start from 5th sign
    part = int((lon % 30.0) // 3.3333333333)  # 0..8
    s = sign_index(lon)
    movable = [0,3,6,9]  # Aries, Cancer, Libra, Capricorn
    fixed = [1,4,7,10]
    dual = [2,5,8,11]
    if s in movable: start = s
    elif s in fixed: start = (s + 8) % 12
    else: start = (s + 4) % 12
    return (start + part) % 12

def d10_dashamsa_sign(lon: float) -> int:
    # Simple rule: each 3-degree segment assigns a sign, pattern differs by odd/even sign
    part = int((lon % 30.0) // 3.0)  # 0..9
    s = sign_index(lon)
    if s % 2 == 0:  # odd sign
        return (s + part) % 12
    else:
        return (s + (9 - part)) % 12

def reinforcement_score(d9_sign_moon: int, d10_sign_mc: int) -> float:
    # Very simple: same element adds reinforcement
    elements = {0:'F',1:'E',2:'A',3:'W'}  # Aries F, Taurus E, Gemini A, Cancer W, etc.
    e9 = elements[d9_sign_moon % 4]
    e10 = elements[d10_sign_mc % 4]
    return 0.15 if e9 == e10 else 0.05
