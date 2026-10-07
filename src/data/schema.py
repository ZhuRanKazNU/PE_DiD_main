"""Table 1 variable definitions and CES-D-10 / exercise-habit instruments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


# Andresen et al. (1994) CES-D-10. Items 5 and 8 are positive affect (reverse-coded).
CESD_ITEMS: List[Dict[str, object]] = [
    {"id": 1, "name": "cesd_1", "reverse": False,
     "en": "I was bothered by things that usually don't bother me.",
     "zh": "一些平时不会困扰我的事情开始让我感到困扰。"},
    {"id": 2, "name": "cesd_2", "reverse": False,
     "en": "I had trouble keeping my mind on what I was doing.",
     "zh": "我很难把注意力集中在正在做的事情上。"},
    {"id": 3, "name": "cesd_3", "reverse": False,
     "en": "I felt depressed.",
     "zh": "我感到情绪低落。"},
    {"id": 4, "name": "cesd_4", "reverse": False,
     "en": "I felt that everything I did was an effort.",
     "zh": "我觉得做任何事都很费力。"},
    {"id": 5, "name": "cesd_5", "reverse": True,
     "en": "I felt hopeful about the future.",
     "zh": "我对未来充满希望。"},
    {"id": 6, "name": "cesd_6", "reverse": False,
     "en": "I felt fearful.",
     "zh": "我感到害怕。"},
    {"id": 7, "name": "cesd_7", "reverse": False,
     "en": "My sleep was restless.",
     "zh": "我的睡眠不安稳。"},
    {"id": 8, "name": "cesd_8", "reverse": True,
     "en": "I was happy.",
     "zh": "我感到快乐。"},
    {"id": 9, "name": "cesd_9", "reverse": False,
     "en": "I felt lonely.",
     "zh": "我感到孤独。"},
    {"id": 10, "name": "cesd_10", "reverse": False,
     "en": "I could not get going.",
     "zh": "我提不起劲来做事。"},
]

CESD_RESPONSE = {
    0: "Rarely or none of the time (less than 1 day)",
    1: "Some or a little of the time (1-2 days)",
    2: "Occasionally or a moderate amount of time (3-4 days)",
    3: "Most or all of the time (5-7 days)",
}

# 14-item, five-point Physical Exercise Habits Scale (Ch.3.4).
# HI is the item-mean (paper: mean/leveled score rather than a single Likert item).
HI_ITEMS: List[Dict[str, object]] = [
    {"id": 1, "name": "hi_1", "en": "I exercise at a regular time each week.",
     "zh": "我每周会在相对固定的时间进行体育锻炼。"},
    {"id": 2, "name": "hi_2", "en": "Physical exercise is part of my daily routine.",
     "zh": "体育锻炼已经成为我日常生活的一部分。"},
    {"id": 3, "name": "hi_3", "en": "I feel uncomfortable if I skip a planned workout.",
     "zh": "如果错过原定锻炼，我会感到不自在。"},
    {"id": 4, "name": "hi_4", "en": "I have a relatively fixed place for exercise.",
     "zh": "我有相对固定的锻炼场所。"},
    {"id": 5, "name": "hi_5", "en": "I keep exercising even when I am busy with coursework.",
     "zh": "即使学业繁忙，我仍会坚持锻炼。"},
    {"id": 6, "name": "hi_6", "en": "My typical workout reaches moderate-to-vigorous intensity.",
     "zh": "我每次锻炼通常能达到中等到较大强度。"},
    {"id": 7, "name": "hi_7", "en": "Each exercise session lasts long enough to be effective.",
     "zh": "我每次锻炼的持续时间足以产生锻炼效果。"},
    {"id": 8, "name": "hi_8", "en": "I have a clear weekly exercise plan.",
     "zh": "我有明确的每周锻炼计划。"},
    {"id": 9, "name": "hi_9", "en": "I enjoy participating in physical exercise.",
     "zh": "我享受参与体育锻炼的过程。"},
    {"id": 10, "name": "hi_10", "en": "I would continue exercising even without PE class requirements.",
     "zh": "即使没有体育课要求，我也会继续锻炼。"},
    {"id": 11, "name": "hi_11", "en": "I monitor my exercise (duration, steps, or heart rate).",
     "zh": "我会记录或关注自己的锻炼情况（时长、步数或心率等）。"},
    {"id": 12, "name": "hi_12", "en": "I often exercise with classmates or teammates.",
     "zh": "我经常与同学或队友一起锻炼。"},
    {"id": 13, "name": "hi_13", "en": "Starting to exercise feels automatic for me.",
     "zh": "对我来说，开始锻炼已经比较自动化。"},
    {"id": 14, "name": "hi_14", "en": "I have maintained this exercise pattern for a long time.",
     "zh": "我已经较长时间保持当前的锻炼习惯。"},
]

HI_RESPONSE = {
    1: "Strongly disagree",
    2: "Disagree",
    3: "Neutral",
    4: "Agree",
    5: "Strongly agree",
}

# 26 survey cities (15 treated + 11 control). First-tier definition follows Ch.3.4:
# Beijing, Shanghai, Guangzhou, Shenzhen, plus commonly classified new first-tier
# cities so that Table 2 UC means (about 0.54-0.58) are attainable.
CITIES = [
    {"city": "Beijing", "first_tier": 1},
    {"city": "Shanghai", "first_tier": 1},
    {"city": "Guangzhou", "first_tier": 1},
    {"city": "Shenzhen", "first_tier": 1},
    {"city": "Chengdu", "first_tier": 1},
    {"city": "Hangzhou", "first_tier": 1},
    {"city": "Chongqing", "first_tier": 1},
    {"city": "Wuhan", "first_tier": 1},
    {"city": "Nanjing", "first_tier": 1},
    {"city": "Tianjin", "first_tier": 1},
    {"city": "Suzhou", "first_tier": 1},
    {"city": "Xi'an", "first_tier": 1},
    {"city": "Changsha", "first_tier": 1},
    {"city": "Zhengzhou", "first_tier": 1},
    {"city": "Qingdao", "first_tier": 0},
    {"city": "Dalian", "first_tier": 0},
    {"city": "Xiamen", "first_tier": 0},
    {"city": "Hefei", "first_tier": 0},
    {"city": "Nanchang", "first_tier": 0},
    {"city": "Fuzhou", "first_tier": 0},
    {"city": "Kunming", "first_tier": 0},
    {"city": "Nanning", "first_tier": 0},
    {"city": "Shijiazhuang", "first_tier": 0},
    {"city": "Taiyuan", "first_tier": 0},
    {"city": "Harbin", "first_tier": 0},
    {"city": "Lanzhou", "first_tier": 0},
]

# Staggered PE-reform adoption among 15 treated cities (time-varying DID).
# Wave index 0-3 corresponds to survey years 2021-2024. Control cities: never treated.
TREATED_ADOPTION = {
    "Beijing": 1,
    "Shanghai": 1,
    "Guangzhou": 1,
    "Chengdu": 1,
    "Hangzhou": 2,
    "Wuhan": 2,
    "Nanjing": 2,
    "Tianjin": 2,
    "Suzhou": 2,
    "Changsha": 3,
    "Zhengzhou": 3,
    "Qingdao": 3,
    "Dalian": 3,
    "Xiamen": 3,
    "Hefei": 3,
}

CONTROL_CITIES = [
    "Shenzhen", "Chongqing", "Xi'an", "Nanchang", "Fuzhou", "Kunming",
    "Nanning", "Shijiazhuang", "Taiyuan", "Harbin", "Lanzhou",
]

EDU_MAP = {
    0: "Illiterate",
    3: "Incomplete primary",
    6: "Primary",
    9: "Junior high",
    12: "Senior high / vocational",
    15: "Junior college",
    16: "Bachelor's degree or above",
}

CONTROLS = ["Gd", "UC", "SH", "AY", "Maj", "FI", "Edu"]
CESD_ITEM_COLS = [it["name"] for it in CESD_ITEMS]
HI_ITEM_COLS = [it["name"] for it in HI_ITEMS]


@dataclass(frozen=True)
class VariableDef:
    vtype: str
    name: str
    symbol: str
    definition: str


TABLE1: List[VariableDef] = [
    VariableDef("Dependent Variable", "Mental Health", "CES-D",
                "CES-D-10 continuous score (0-30); higher = more depressive symptoms."),
    VariableDef("Key Explanatory Variable", "College PE Reform", "time x treated",
                "Equals 1 if the student's college has implemented the PE reform in period t; 0 otherwise."),
    VariableDef("Mediating Variable", "Exercise Habit Intensity", "HI",
                "14-item, five-point Likert scale (1-5); higher = more stable exercise habits."),
    VariableDef("Mediating Variable", "Exercise Frequency", "EF",
                "The number of physical exercises per week."),
    VariableDef("Control Variable", "Gender", "Gd", "Male = 1, female = 0."),
    VariableDef("Control Variable", "College city", "UC",
                "First-tier city in China = 1; otherwise = 0."),
    VariableDef("Control Variable", "Self-rated Health", "SH",
                "Five-point scale (1-5); original item asked about mental health (interpret with caution)."),
    VariableDef("Control Variable", "Academic Year", "AY", "First to fourth year (1 to 4)."),
    VariableDef("Control Variable", "Major Type", "Maj",
                "Science and engineering = 1, humanities and social sciences = 0."),
    VariableDef("Control Variable", "Family Economic Status", "FI",
                "Five-point Likert scale (1-5)."),
    VariableDef("Control Variable", "Parental Education", "Edu",
                "Illiterate = 0, incomplete primary = 3, primary = 6, junior high = 9, "
                "senior high/vocational = 12, junior college = 15, bachelor's degree or above = 16."),
]
