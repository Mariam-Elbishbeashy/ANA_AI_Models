#!/usr/bin/env python3
"""
HIGH-QUALITY IFS Character Dataset Generator - ENHANCED VERSION
Reads raw text datasets, preprocesses them, and generates realistic sentences
with no duplication
"""

import os
import sys
import csv
import json
import argparse
import random
import re
import time
import hashlib
import glob
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from pathlib import Path

# ---------------- Configuration ----------------
SEED = 12345
random.seed(SEED)

# Define as a variable that can be modified
RAW_DATA_DIR = "data/raw"
OUT_DIR_DEFAULT = "data/processed"
TOTAL_EXAMPLES = 360000  # 20,000 per character × 18 characters
EXAMPLES_PER_CHARACTER = 20000

# ---------------- All Characters ----------------
ALL_CHARACTERS = [
    "Inner Critic", "Perfectionist", "People-Pleaser", "Controller", "Stoic Part",
    "Workaholic", "Confused Part", "Procrastinator", "Overeater/Binger", "Excessive Gamer",
    "Lonely Part", "Fearful Part", "Neglected Part", "Ashamed Part", "Overwhelmed Part",
    "Dependent Part", "Jealous Part", "Wounded Child"
]

# ---------------- Character Core Patterns ----------------
# These are the fundamental patterns for each character
# We'll enhance these with patterns from raw data

CHARACTER_PATTERNS = {
    "Inner Critic": {
        "emotion": "sadness",
        "category": "critical",
        "core_phrases": [
            "not good enough", "failure", "worthless", "disappointment", "inadequate",
            "never good enough", "always failing", "can't do anything right", "burden",
            "useless", "hopeless", "incompetent", "flawed", "sabotage myself",
            "worst enemy", "don't deserve", "unworthy", "what's wrong with me",
            "others are better", "fundamentally flawed", "mess everything up"
        ],
        "templates": [
            "I feel like I'm {}",
            "I'm always {}",
            "No matter what I do, I feel {}",
            "My inner voice tells me I'm {}",
            "I can't shake the feeling that I'm {}",
            "Deep down, I believe I'm {}",
            "I constantly worry that I'm {}",
            "The thought that I'm {} never leaves me",
            "I'm struggling with feeling {}",
            "I've been told I'm {} for so long that I believe it"
        ]
    },
    
    "Perfectionist": {
        "emotion": "fear",
        "category": "anxious",
        "core_phrases": [
            "has to be perfect", "flawless", "no mistakes", "exactly right", "every detail",
            "redo everything", "good enough isn't", "perfection or nothing", "high standards",
            "can't accept imperfect", "obsess over details", "must be flawless", "terrified of errors",
            "nothing is ever good enough", "rework constantly", "precision matters", "zero tolerance for errors"
        ],
        "templates": [
            "Everything {}",
            "I need everything to be {}",
            "I can't submit work unless it's {}",
            "The thought of {} gives me anxiety",
            "I spend hours making sure things are {}",
            "My work has to be {} or I won't share it",
            "I'm constantly worried about {}",
            "{} is non-negotiable for me",
            "I can't relax until things are {}",
            "{} controls my entire process"
        ]
    },
    
    "People-Pleaser": {
        "emotion": "fear",
        "category": "anxious",
        "core_phrases": [
            "say yes", "avoid conflict", "need approval", "please others", "people's needs",
            "don't want to upset", "want everyone to like me", "can't say no", "put others first",
            "responsible for feelings", "fear rejection", "sacrifice my needs", "keep peace",
            "change myself to fit in", "avoid disagreement", "need validation", "prioritize others",
            "afraid of disapproval", "cater to everyone", "lose myself in others"
        ],
        "templates": [
            "I always {}",
            "I can't stop myself from {}",
            "My need to {} controls my life",
            "I'm terrified of {}",
            "{} is my default mode",
            "I feel compelled to {}",
            "My biggest fear is {}",
            "I've built my life around {}",
            "{} feels like survival to me",
            "I don't know how to stop {}"
        ]
    },
    
    "Controller": {
        "emotion": "anger",
        "category": "protective",
        "core_phrases": [
            "in control", "my way", "micromanage", "need structure", "oversee everything",
            "losing control terrifies", "chaos results", "can't delegate", "need to manage",
            "everything must follow my plan", "spontaneity uncomfortable", "unpredictability dangerous",
            "must supervise", "things fall apart without me", "trust issues", "need order",
            "system works best", "anxious when not in charge", "prefer predictability"
        ],
        "templates": [
            "I need to be {}",
            "{} is essential for me",
            "When I'm not {}, I panic",
            "My need to {} affects my relationships",
            "{} feels like safety to me",
            "I can't function without {}",
            "{} is how I cope with uncertainty",
            "My insistence on {} causes problems",
            "I feel most secure when {}",
            "{} has become my coping mechanism"
        ]
    },
    
    "Stoic Part": {
        "emotion": "neutral",
        "category": "protective",
        "core_phrases": [
            "don't show emotions", "emotional control", "detached", "composed", "unemotional",
            "keep feelings inside", "emotions are weak", "stay calm", "suppress emotions",
            "rational not emotional", "feelings get in the way", "maintain composure", "emotional distance",
            "don't express feelings", "keep a poker face", "emotional walls", "avoid vulnerability"
        ],
        "templates": [
            "I rarely {}",
            "{} is my natural state",
            "People tell me I'm too {}",
            "I've learned to {}",
            "{} protects me from getting hurt",
            "My tendency to {} isolates me",
            "I feel safest when I'm {}",
            "{} has become my defense mechanism",
            "I don't know how to stop {}",
            "{} feels like my only option"
        ]
    },
    
    "Workaholic": {
        "emotion": "neutral",
        "category": "compulsive",
        "core_phrases": [
            "always working", "too busy", "productive", "can't stop working", "work identity",
            "worth tied to work", "overworking", "defined by work", "rest feels guilty",
            "achievement focused", "constant productivity", "work comes first", "thrive on busyness",
            "fear of unproductivity", "using work to avoid", "workaholic tendencies", "neglect self-care"
        ],
        "templates": [
            "I'm constantly {}",
            "{} defines who I am",
            "My need to {} consumes me",
            "I can't imagine life without {}",
            "{} is how I prove my worth",
            "I use {} to avoid other issues",
            "My identity is tied to {}",
            "{} feels like my purpose",
            "I feel lost when I'm not {}",
            "{} has taken over my life"
        ]
    },
    
    "Confused Part": {
        "emotion": "fear",
        "category": "disoriented",
        "core_phrases": [
            "confused", "don't understand", "mixed up", "unclear", "indecisive",
            "mind is foggy", "can't think straight", "everything unclear", "lost",
            "uncertain", "disoriented", "mental fog", "lack clarity", "bewildered",
            "puzzled", "torn between options", "can't make sense", "overwhelmed by choices"
        ],
        "templates": [
            "I feel constantly {}",
            "{} is my default state",
            "My mind is always {}",
            "I struggle with feeling {}",
            "{} makes everyday decisions difficult",
            "I can't escape this feeling of being {}",
            "{} affects everything I do",
            "My constant {} holds me back",
            "I wish I could stop feeling {}",
            "{} clouds my judgment"
        ]
    },
    
    "Procrastinator": {
        "emotion": "neutral",
        "category": "avoidant",
        "core_phrases": [
            "do it later", "tomorrow", "put off", "delaying", "last minute",
            "avoiding starting", "not in the mood", "wait for motivation", "work under pressure",
            "plenty of time", "just one more", "building up to it", "avoid tasks", "delay action",
            "procrastination habit", "wait for right moment", "future me will handle"
        ],
        "templates": [
            "I always {}",
            "My habit of {} causes problems",
            "I tell myself I'll {}",
            "{} is my coping mechanism",
            "I can't seem to stop {}",
            "My tendency to {} affects my life",
            "{} feels easier in the moment",
            "I know I should stop {}",
            "{} has become a pattern",
            "I use {} to avoid discomfort"
        ]
    },
    
    "Overeater/Binger": {
        "emotion": "sadness",
        "category": "compulsive",
        "core_phrases": [
            "can't stop eating", "emotional eating", "binge", "food addiction", "eating when stressed",
            "using food to cope", "food comforts", "eat to numb", "secret eating", "guilty after eating",
            "shame about food", "out of control with food", "stuff feelings with food", "numbing with food",
            "food as escape", "compulsive eating", "eat when emotional", "food relationship issues"
        ],
        "templates": [
            "When I'm stressed, I {}",
            "I use food to {}",
            "My pattern of {} shames me",
            "{} is how I cope with emotions",
            "I feel out of control when I {}",
            "{} provides temporary relief",
            "My relationship with food involves {}",
            "I'm trying to stop {}",
            "{} feels like my only comfort",
            "I struggle with {} constantly"
        ]
    },
    
    "Excessive Gamer": {
        "emotion": "sadness",
        "category": "compulsive",
        "core_phrases": [
            "gaming all night", "escaping through games", "virtual world", "gaming addiction", "obsessed",
            "avoiding real world", "digital escape", "prefer virtual life", "game rewards", "gamer identity",
            "reality avoidance", "using games to cope", "can't stop gaming", "games over responsibilities",
            "virtual success", "gaming community", "leveling up everything"
        ],
        "templates": [
            "I spend hours {}",
            "{} helps me escape reality",
            "My habit of {} worries me",
            "I use gaming to {}",
            "{} feels better than real life",
            "My tendency to {} affects my responsibilities",
            "{} provides what real life doesn't",
            "I'm concerned about my {}",
            "{} has become my main coping method",
            "I feel most alive when I'm {}"
        ]
    },
    
    "Lonely Part": {
        "emotion": "sadness",
        "category": "sad",
        "core_phrases": [
            "lonely", "alone", "isolated", "disconnected", "no one understands",
            "unwanted", "abandoned", "left out", "empty inside", "yearning for connection",
            "feeling separate", "emotional isolation", "craving companionship", "unloved",
            "on the outside", "missing connection", "void inside", "socially isolated"
        ],
        "templates": [
            "I feel deeply {}",
            "This sense of {} never leaves",
            "My constant {} affects my happiness",
            "I'm struggling with feeling {}",
            "{} colors my entire experience",
            "The feeling of being {} is overwhelming",
            "I can't escape this {}",
            "My life feels defined by {}",
            "{} makes social situations difficult",
            "I wish I could stop feeling so {}"
        ]
    },
    
    "Fearful Part": {
        "emotion": "fear",
        "category": "anxious",
        "core_phrases": [
            "scared", "afraid", "worried", "anxious", "panic",
            "worst-case", "danger", "unsafe", "vulnerable", "fear of failure",
            "anticipatory anxiety", "terrified", "what if something bad", "apprehensive",
            "constant worry", "catastrophic thinking", "risk averse", "fearful mindset"
        ],
        "templates": [
            "I'm constantly {}",
            "My mind always goes to {}",
            "{} controls my decisions",
            "I live in a state of {}",
            "My tendency to {} limits me",
            "{} feels like my constant companion",
            "I can't escape feeling {}",
            "My life is ruled by {}",
            "{} prevents me from taking risks",
            "I wish I could stop being so {}"
        ]
    },
    
    "Neglected Part": {
        "emotion": "sadness",
        "category": "sad",
        "core_phrases": [
            "neglected", "ignored", "unnoticed", "overlooked", "unimportant",
            "invisible", "unseen", "unheard", "attention seeking", "emotional neglect",
            "unmet needs", "passed over", "forgotten", "unattended", "yearning for attention",
            "feel insignificant", "background person", "never the priority"
        ],
        "templates": [
            "I often feel {}",
            "This feeling of being {} hurts",
            "My experience of {} affects my self-worth",
            "I struggle with feeling {}",
            "{} has been a pattern in my life",
            "The sense of being {} is painful",
            "I can't shake this feeling of being {}",
            "{} makes it hard to ask for what I need",
            "My childhood involved a lot of {}",
            "I'm working through feelings of {}"
        ]
    },
    
    "Ashamed Part": {
        "emotion": "sadness",
        "category": "shame",
        "core_phrases": [
            "ashamed", "shame", "embarrassed", "guilty", "flawed",
            "unworthy", "bad person", "moral failure", "dirty secret", "humiliated",
            "regretful", "remorseful", "self-conscious", "judged", "defective",
            "undeserving", "stained", "hidden shame", "core shame"
        ],
        "templates": [
            "I carry deep {}",
            "This feeling of {} never leaves",
            "My sense of {} affects everything",
            "I'm struggling with {}",
            "{} colors how I see myself",
            "The weight of {} is heavy",
            "I can't escape this {}",
            "{} makes it hard to connect",
            "My life has been marked by {}",
            "I'm trying to heal from {}"
        ]
    },
    
    "Overwhelmed Part": {
        "emotion": "fear",
        "category": "anxious",
        "core_phrases": [
            "overwhelmed", "too much", "can't handle", "drowning", "pressure",
            "stress", "burnout", "exhausted", "sensory overload", "multiple demands",
            "flooded", "swamped", "buried", "crushed", "information overload",
            "emotional overload", "pulled in directions", "no space", "need break"
        ],
        "templates": [
            "I feel constantly {}",
            "This state of {} is exhausting",
            "My tendency to get {} affects my health",
            "I'm struggling with feeling {}",
            "{} makes everyday tasks difficult",
            "The feeling of being {} never stops",
            "I can't seem to stop feeling {}",
            "{} prevents me from functioning well",
            "My life feels defined by {}",
            "I'm trying to manage this constant {}"
        ]
    },
    
    "Dependent Part": {
        "emotion": "fear",
        "category": "anxious",
        "core_phrases": [
            "can't do alone", "need help", "dependent", "rely on others", "helpless",
            "co-dependent", "need others", "fear independence", "clingy", "need guidance",
            "incapable alone", "enmeshed", "fused", "no autonomy", "attachment issues",
            "need validation", "seek approval", "look to others", "follow others"
        ],
        "templates": [
            "I struggle with {}",
            "My need to {} concerns me",
            "I feel {} in relationships",
            "My pattern of {} holds me back",
            "{} affects my independence",
            "I can't seem to stop {}",
            "My tendency to {} worries me",
            "{} makes it hard to trust myself",
            "I'm trying to overcome {}",
            "{} has been a lifelong pattern"
        ]
    },
    
    "Jealous Part": {
        "emotion": "anger",
        "category": "protective",
        "core_phrases": [
            "jealous", "envious", "resentful", "comparison", "others have what i want",
            "why them not me", "possessive", "fear of loss", "inadequate compared", "life unfair",
            "covet what others have", "green-eyed", "territorial", "suspicious", "distrustful",
            "competitive", "rivalry", "wish i had", "bitter about success"
        ],
        "templates": [
            "I feel {} when others succeed",
            "My tendency to {} shames me",
            "I struggle with feelings of {}",
            "{} affects my relationships",
            "I can't seem to stop feeling {}",
            "My pattern of {} worries me",
            "{} makes it hard to be happy for others",
            "I'm working on my {}",
            "{} feels like a reflex",
            "My feelings of {} concern me"
        ]
    },
    
    "Wounded Child": {
        "emotion": "sadness",
        "category": "vulnerable",
        "core_phrases": [
            "inner child", "childhood wound", "past hurt", "emotional wound", "trauma",
            "vulnerable child", "abandoned child", "rejected child", "hurt child", "core wound",
            "younger self", "needy child", "frightened child", "lonely child", "shamed child",
            "early wound", "developmental trauma", "attachment wound", "unhealed pain"
        ],
        "templates": [
            "My {} still affects me",
            "I carry this {} with me",
            "The {} from my past influences present",
            "I'm healing from {}",
            "My {} needs attention",
            "This {} shapes how I relate",
            "I'm learning to parent my {}",
            "The pain of my {} surfaces often",
            "My {} feels raw sometimes",
            "I'm working with my {}"
        ]
    }
}

# ---------------- Variation System ----------------
PREFIXES = [
    "Honestly, ", "Sometimes ", "Lately, ", "Often ", "Recently, ",
    "The truth is, ", "I've noticed that ", "I feel like ", "It seems like ",
    "In my experience, ", "For me, ", "Personally, ", "To be honest, ",
    "I have to admit, ", "I realize that ", "I'm learning that ",
    "More and more, ", "Increasingly, ", "These days, ", "Right now, "
]

SUFFIXES = [
    ".", " and it's really hard.", " but I'm working on it.",
    " which affects my daily life.", " and I don't know what to do about it.",
    " and it's exhausting.", " but I'm trying to change.",
    " and it impacts my relationships.", " even though I wish it weren't true.",
    " and I'm not sure how to fix it.", " which makes everything more difficult.",
    " and it's taking a toll on me.", " but change feels impossible.",
    " and I'm tired of feeling this way.", " which is why I'm seeking help.",
    " and it's become a pattern.", " but awareness is the first step.",
    " and I'm learning to accept it.", " which requires constant attention.",
    " and I'm developing strategies to cope."
]

CONTEXTS = [
    "When I'm at work, ", "In relationships, ", "With my family, ",
    "Around friends, ", "When I'm alone, ", "During stressful times, ",
    "When I feel triggered, ", "In social situations, ", "When facing challenges, ",
    "In my daily life, ", "When making decisions, ", "In new situations, ",
    "When things get difficult, ", "During conflicts, ", "When I'm tired, ",
    "When expectations are high, ", "In unfamiliar environments, ", "When I need to perform, ",
    "When others depend on me, ", "When I'm under pressure, "
]

INTENSIFIERS = [
    "deeply", "constantly", "always", "never", "constantly",
    "overwhelmingly", "profoundly", "intensely", "persistently",
    "chronically", "habitually", "automatically", "reflexively",
    "unconsciously", "instinctively", "inescapably", "pervasively"
]

# ---------------- Helper Functions ----------------
def clean_text(text: str) -> str:
    """Clean text while preserving meaning."""
    if not text:
        return ""
    text = text.strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Ensure proper sentence structure
    if not text.endswith(('.', '!', '?')):
        text = text + '.'
    return text

def load_raw_datasets(raw_data_dir: str) -> Dict[str, List[str]]:
    """
    Load all text datasets from the raw data directory.
    Returns a dictionary mapping dataset names to lists of sentences.
    """
    raw_data = {}
    
    if not os.path.exists(raw_data_dir):
        print(f"⚠️ Warning: Raw data directory '{raw_data_dir}' not found.")
        print(f"   Creating synthetic data instead.")
        return raw_data
    
    # Find all text files
    text_files = []
    for ext in ['*.txt', '*.csv', '*.json', '*.jsonl']:
        text_files.extend(glob.glob(os.path.join(raw_data_dir, '**', ext), recursive=True))
        text_files.extend(glob.glob(os.path.join(raw_data_dir, ext)))
    
    print(f"📂 Found {len(text_files)} raw data files")
    
    for file_path in text_files:
        filename = os.path.basename(file_path)
        try:
            if filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Split into sentences
                    sentences = re.split(r'[.!?]+', content)
                    sentences = [clean_text(s) for s in sentences if clean_text(s)]
                    if sentences:
                        raw_data[filename] = sentences
                        print(f"   Loaded {len(sentences)} sentences from {filename}")
            
            elif filename.endswith('.csv'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    sentences = []
                    for row in reader:
                        if row:
                            # Take first column or join all columns
                            text = row[0] if len(row) > 0 else ' '.join(row)
                            if text and len(text.strip()) > 10:  # Minimum length
                                sentences.append(clean_text(text))
                    if sentences:
                        raw_data[filename] = sentences
                        print(f"   Loaded {len(sentences)} rows from {filename}")
            
            elif filename.endswith('.json') or filename.endswith('.jsonl'):
                sentences = []
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    if filename.endswith('.jsonl'):
                        for line in f:
                            try:
                                data = json.loads(line.strip())
                                if isinstance(data, dict) and 'text' in data:
                                    sentences.append(clean_text(data['text']))
                                elif isinstance(data, str):
                                    sentences.append(clean_text(data))
                            except:
                                pass
                    else:
                        try:
                            data = json.load(f)
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict) and 'text' in item:
                                        sentences.append(clean_text(item['text']))
                                    elif isinstance(item, str):
                                        sentences.append(clean_text(item))
                            elif isinstance(data, dict):
                                for key, value in data.items():
                                    if isinstance(value, str) and len(value) > 10:
                                        sentences.append(clean_text(value))
                        except:
                            pass
                
                if sentences:
                    raw_data[filename] = sentences
                    print(f"   Loaded {len(sentences)} items from {filename}")
        
        except Exception as e:
            print(f"   Error loading {filename}: {str(e)}")
    
    # Combine all sentences into one large list
    all_sentences = []
    for sentences in raw_data.values():
        all_sentences.extend(sentences)
    
    print(f"\n📊 Total raw sentences loaded: {len(all_sentences)}")
    return {"all_sentences": all_sentences, **raw_data}

def extract_patterns_from_text(text: str, character: str) -> List[str]:
    """
    Extract relevant patterns and phrases from raw text for a specific character.
    """
    patterns = []
    
    # Define character-specific keywords
    character_keywords = {
        "Inner Critic": ["fail", "worthless", "inadequate", "not good", "disappoint", "burden"],
        "Perfectionist": ["perfect", "flawless", "mistake", "detail", "high standard", "redo"],
        "People-Pleaser": ["please", "approval", "conflict", "say yes", "rejection", "others first"],
        "Controller": ["control", "micromanage", "structure", "plan", "order", "chaos"],
        "Stoic Part": ["emotion", "detached", "composed", "suppress", "rational", "unemotional"],
        "Workaholic": ["work", "busy", "productive", "achieve", "overwork", "rest guilt"],
        "Confused Part": ["confused", "unclear", "indecisive", "foggy", "uncertain", "lost"],
        "Procrastinator": ["later", "tomorrow", "delay", "put off", "last minute", "avoid"],
        "Overeater/Binger": ["eat", "food", "binge", "emotional eating", "numb", "comfort food"],
        "Excessive Gamer": ["game", "gaming", "virtual", "escape", "addict", "obsess"],
        "Lonely Part": ["lonely", "alone", "isolated", "disconnect", "unwanted", "empty"],
        "Fearful Part": ["scared", "afraid", "anxious", "worry", "panic", "fear"],
        "Neglected Part": ["neglect", "ignore", "unseen", "unheard", "invisible", "overlook"],
        "Ashamed Part": ["ashamed", "shame", "guilty", "embarrassed", "unworthy", "flawed"],
        "Overwhelmed Part": ["overwhelm", "too much", "stress", "burnout", "exhaust", "pressure"],
        "Dependent Part": ["dependent", "rely", "help", "cling", "need others", "independent"],
        "Jealous Part": ["jealous", "envious", "resent", "compare", "possessive", "rivalry"],
        "Wounded Child": ["child", "wound", "trauma", "hurt", "abandon", "reject"]
    }
    
    keywords = character_keywords.get(character, [])
    
    # Convert to lowercase for matching
    text_lower = text.lower()
    
    # Find sentences containing keywords
    for keyword in keywords:
        if keyword in text_lower:
            # Extract phrase around keyword
            words = text.split()
            for i, word in enumerate(words):
                if keyword in word.lower():
                    # Get context window around keyword
                    start = max(0, i - 3)
                    end = min(len(words), i + 4)
                    phrase = ' '.join(words[start:end])
                    patterns.append(phrase)
                    break
    
    return patterns

def enhance_character_patterns(raw_data: Dict[str, List[str]]) -> Dict:
    """
    Enhance character patterns with patterns extracted from raw data.
    """
    enhanced_patterns = CHARACTER_PATTERNS.copy()
    
    all_sentences = raw_data.get("all_sentences", [])
    if not all_sentences:
        print("⚠️ No raw sentences found, using base patterns only")
        return enhanced_patterns
    
    print("\n🔍 Extracting patterns from raw data...")
    
    for character in ALL_CHARACTERS:
        character_patterns = []
        
        # Extract patterns from raw sentences
        for sentence in all_sentences:
            patterns = extract_patterns_from_text(sentence, character)
            character_patterns.extend(patterns)
        
        # Remove duplicates and clean
        character_patterns = list(set(character_patterns))
        character_patterns = [clean_text(p) for p in character_patterns if len(clean_text(p)) > 10]
        
        if character_patterns:
            # Add to core phrases (limit to 10 additional patterns)
            additional_patterns = character_patterns[:10]
            enhanced_patterns[character]["core_phrases"].extend(additional_patterns)
            
            # Create new templates from raw sentences (if they contain patterns)
            new_templates = []
            for pattern in additional_patterns[:5]:  # Use first 5 patterns for templates
                # Create template by replacing pattern with placeholder
                if ' ' in pattern:
                    words = pattern.split()
                    if len(words) > 2:
                        # Try to create a template
                        template = pattern
                        for keyword in ["feel", "am", "have", "do", "need", "want"]:
                            if keyword in pattern.lower():
                                parts = pattern.split(keyword, 1)
                                if len(parts) > 1:
                                    template = f"I {keyword} {{}}" + parts[1]
                                    break
                        new_templates.append(template)
            
            if new_templates:
                enhanced_patterns[character]["templates"].extend(new_templates[:3])  # Add up to 3 new templates
            
            print(f"   {character}: Added {len(additional_patterns)} patterns, {len(new_templates)} templates")
    
    return enhanced_patterns

def create_realistic_variation(base_template: str, core_phrase: str, 
                              raw_sentences: List[str], variation_num: int) -> str:
    """
    Create a realistic variation using raw sentence patterns.
    """
    rng = random.Random(SEED + hash(base_template) + hash(core_phrase) + variation_num)
    
    # 30% chance to use raw sentence structure instead of template
    if raw_sentences and rng.random() < 0.3:
        raw_sentence = rng.choice(raw_sentences)
        if core_phrase in raw_sentence.lower():
            # Replace part of raw sentence with core phrase
            text = raw_sentence
        else:
            # Insert core phrase into raw sentence
            words = raw_sentence.split()
            if len(words) > 3:
                insert_pos = rng.randint(1, len(words) - 2)
                words.insert(insert_pos, core_phrase)
                text = ' '.join(words)
            else:
                text = base_template.format(core_phrase)
    else:
        text = base_template.format(core_phrase)
    
    # Apply random modifications
    modifications = []
    
    # Add prefix (30% chance)
    if rng.random() < 0.3:
        prefix = rng.choice(PREFIXES)
        modifications.append(("prefix", prefix))
    
    # Add suffix (40% chance)
    if rng.random() < 0.4:
        suffix = rng.choice(SUFFIXES)
        modifications.append(("suffix", suffix))
    
    # Add context (25% chance)
    if rng.random() < 0.25:
        context = rng.choice(CONTEXTS)
        modifications.append(("context", context))
    
    # Add intensifier (20% chance)
    if rng.random() < 0.2 and " " in core_phrase:
        words = core_phrase.split()
        if len(words) > 1:
            intensifier = rng.choice(INTENSIFIERS)
            # Insert intensifier before adjective or after first word
            modified_phrase = f"{words[0]} {intensifier} {' '.join(words[1:])}"
            text = base_template.format(modified_phrase)
    
    # Apply modifications in random order
    rng.shuffle(modifications)
    
    for mod_type, mod_text in modifications:
        if mod_type == "prefix":
            text = mod_text + text[0].lower() + text[1:] if text else text
        elif mod_type == "suffix":
            text = text.rstrip('.!?') + mod_text
        elif mod_type == "context":
            text = mod_text + text[0].lower() + text[1:] if text else text
    
    return clean_text(text)

def create_compound_variation(char1_data: dict, char2_data: dict, 
                             raw_sentences: List[str], variation_num: int) -> str:
    """Create a realistic compound variation combining two character patterns."""
    rng = random.Random(SEED + variation_num * 1000)
    
    # 20% chance to use raw sentence for compound
    if raw_sentences and rng.random() < 0.2:
        raw_sentence = rng.choice(raw_sentences)
        # Check if raw sentence contains patterns from both characters
        char1_phrases = [p for p in char1_data["core_phrases"] if p in raw_sentence.lower()]
        char2_phrases = [p for p in char2_data["core_phrases"] if p in raw_sentence.lower()]
        
        if char1_phrases and char2_phrases:
            return clean_text(raw_sentence)
    
    # Get random elements from each character
    phrase1 = rng.choice(char1_data["core_phrases"])
    phrase2 = rng.choice(char2_data["core_phrases"])
    
    # Create compound patterns with more natural language
    patterns = [
        f"I notice that when I {phrase1}, I also tend to {phrase2}",
        f"Somehow my {phrase1} and {phrase2} feel connected",
        f"It seems like {phrase1} and {phrase2} go hand in hand for me",
        f"I'm starting to see how my {phrase1} relates to my {phrase2}",
        f"When I feel {phrase1}, it often brings up {phrase2}",
        f"My therapist pointed out the link between my {phrase1} and {phrase2}",
        f"I've noticed a pattern: when I'm {phrase1}, I'm also more likely to {phrase2}",
        f"The connection between {phrase1} and {phrase2} is becoming clearer to me",
        f"It's hard to separate my {phrase1} from my {phrase2}",
        f"Some days, my {phrase1} and {phrase2} feel intertwined"
    ]
    
    text = rng.choice(patterns)
    
    # Apply random modifications
    if rng.random() < 0.3:
        text = rng.choice(PREFIXES) + text[0].lower() + text[1:]
    if rng.random() < 0.4:
        text = text.rstrip('.!?') + rng.choice(SUFFIXES)
    
    return clean_text(text)

def generate_character_cues(character: str) -> str:
    """Generate character cues string."""
    if character in CHARACTER_PATTERNS:
        phrases = CHARACTER_PATTERNS[character]["core_phrases"][:10]  # First 10 phrases
        return "; ".join(phrases)
    return character

# ---------------- Main Generation Function ----------------
def generate_high_quality_dataset(raw_data_dir: str, out_dir: str, examples_per_char: int = EXAMPLES_PER_CHARACTER):
    """Generate high-quality dataset with no duplicates and perfect matching."""
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "ifs_high_quality_dataset.csv")
    
    print("=" * 70)
    print("🎯 GENERATING HIGH-QUALITY IFS DATASET")
    print("=" * 70)
    
    # Load raw data
    print("📂 Loading raw datasets...")
    raw_data = load_raw_datasets(raw_data_dir)
    
    # Enhance character patterns with raw data
    print("\n🔧 Enhancing character patterns with raw data...")
    enhanced_patterns = enhance_character_patterns(raw_data)
    
    all_sentences = raw_data.get("all_sentences", [])
    
    print(f"\n📊 Dataset Configuration:")
    print(f"   Characters: {len(ALL_CHARACTERS)}")
    print(f"   Examples per character: {examples_per_char:,}")
    print(f"   Total examples: {examples_per_char * len(ALL_CHARACTERS):,}")
    print(f"   Raw sentences available: {len(all_sentences)}")
    print("=" * 70)
    
    start_time = time.time()
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "input_text", "detected_emotion", "inner_char", "character_cues", "category"])
        
        row_id = 1
        total_generated = 0
        global_text_set = set()
        
        for character_idx, character in enumerate(ALL_CHARACTERS):
            char_start_time = time.time()
            print(f"\n📝 Processing: {character}")
            
            char_data = enhanced_patterns[character]
            emotion = char_data["emotion"]
            category = char_data["category"]
            cues = generate_character_cues(character)
            
            templates = char_data["templates"]
            core_phrases = char_data["core_phrases"]
            
            char_text_set = set()
            examples_generated = 0
            
            # Phase 1: Basic template variations (60% of examples)
            print(f"  Phase 1: Basic variations...")
            basic_target = int(examples_per_char * 0.6)
            
            template_idx = 0
            phrase_idx = 0
            variation_num = 0
            
            while examples_generated < basic_target:
                # Cycle through templates and phrases
                template = templates[template_idx % len(templates)]
                phrase = core_phrases[phrase_idx % len(core_phrases)]
                
                # Generate realistic variation
                text = create_realistic_variation(template, phrase, all_sentences, variation_num)
                text_hash = hashlib.md5(text.encode()).hexdigest()
                
                # Check for duplicates
                if text_hash not in global_text_set and text_hash not in char_text_set:
                    global_text_set.add(text_hash)
                    char_text_set.add(text_hash)
                    
                    writer.writerow([row_id, text, emotion, character, cues, category])
                    row_id += 1
                    examples_generated += 1
                    total_generated += 1
                    
                    if examples_generated % 1000 == 0:
                        print(f"    Generated {examples_generated:,}/{basic_target:,} basic examples")
                        f.flush()
                
                variation_num += 1
                template_idx += 1
                phrase_idx += 1
                
                # If we're stuck, move to next phase
                if variation_num > 100000:  # Safety check
                    print(f"    Warning: Reached variation limit, moving to next phase")
                    break
            
            # Phase 2: Compound variations with other characters (25% of examples)
            print(f"  Phase 2: Compound variations...")
            compound_target = int(examples_per_char * 0.25)
            
            other_chars = [c for c in ALL_CHARACTERS if c != character]
            compound_variation_num = 0
            
            while examples_generated < (basic_target + compound_target):
                # Pick another character at random
                other_char = random.choice(other_chars)
                other_data = enhanced_patterns[other_char]
                
                # Generate compound variation
                text = create_compound_variation(char_data, other_data, all_sentences, compound_variation_num)
                text_hash = hashlib.md5(text.encode()).hexdigest()
                
                # Check for duplicates
                if text_hash not in global_text_set and text_hash not in char_text_set:
                    global_text_set.add(text_hash)
                    char_text_set.add(text_hash)
                    
                    writer.writerow([row_id, text, emotion, character, cues, category])
                    row_id += 1
                    examples_generated += 1
                    total_generated += 1
                    
                    if examples_generated % 1000 == 0:
                        print(f"    Generated {examples_generated - basic_target:,}/{compound_target:,} compound examples")
                        f.flush()
                
                compound_variation_num += 1
                
                # Safety check
                if compound_variation_num > 50000:
                    print(f"    Warning: Reached compound variation limit")
                    break
            
            # Phase 3: Advanced variations (remaining 15%)
            print(f"  Phase 3: Advanced variations...")
            advanced_target = examples_per_char
            advanced_variation_num = 0
            
            # More complex sentence structures inspired by raw data
            advanced_patterns = [
                "What I'm realizing is that {} and this awareness is changing how I approach things",
                "My journey with {} has taught me important lessons about myself",
                "Working through {} has been challenging but transformative",
                "I'm beginning to understand the roots of my {}",
                "The pattern of {} emerges most strongly when {}",
                "My therapist says that my {} comes from {}",
                "I notice that {} happens whenever I feel {}",
                "Learning to manage my {} has been a key part of my growth",
                "The more I heal my {}, the less I experience {}",
                "{} used to control my life, but now I'm learning to {}"
            ]
            
            while examples_generated < advanced_target:
                # Use a random advanced pattern
                pattern = random.choice(advanced_patterns)
                
                # Fill with appropriate content
                if pattern.count("{}") == 1:
                    phrase = random.choice(core_phrases)
                    text = pattern.format(phrase)
                elif pattern.count("{}") == 2:
                    phrase1 = random.choice(core_phrases)
                    phrase2 = random.choice(core_phrases)
                    text = pattern.format(phrase1, phrase2)
                else:
                    text = pattern
                
                # Apply modifications
                rng = random.Random(SEED + advanced_variation_num)
                if rng.random() < 0.3:
                    text = rng.choice(PREFIXES) + text[0].lower() + text[1:]
                if rng.random() < 0.4:
                    text = text.rstrip('.!?') + rng.choice(SUFFIXES)
                
                text = clean_text(text)
                text_hash = hashlib.md5(text.encode()).hexdigest()
                
                # Check for duplicates
                if text_hash not in global_text_set and text_hash not in char_text_set:
                    global_text_set.add(text_hash)
                    char_text_set.add(text_hash)
                    
                    writer.writerow([row_id, text, emotion, character, cues, category])
                    row_id += 1
                    examples_generated += 1
                    total_generated += 1
                    
                    if examples_generated % 1000 == 0:
                        print(f"    Generated {examples_generated:,}/{examples_per_char:,} total examples")
                        f.flush()
                
                advanced_variation_num += 1
                
                # Safety check
                if advanced_variation_num > 30000 and examples_generated < advanced_target:
                    # Generate simple variations to reach target
                    print(f"    Generating simple variations to reach target...")
                    while examples_generated < advanced_target:
                        template = random.choice(templates)
                        phrase = random.choice(core_phrases)
                        text = create_realistic_variation(template, phrase, all_sentences, advanced_variation_num)
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        
                        if text_hash not in global_text_set:
                            global_text_set.add(text_hash)
                            char_text_set.add(text_hash)
                            
                            writer.writerow([row_id, text, emotion, character, cues, category])
                            row_id += 1
                            examples_generated += 1
                            total_generated += 1
                            advanced_variation_num += 1
                    
                    break
            
            char_time = time.time() - char_start_time
            print(f"  ✅ {character}: {examples_generated:,} examples in {char_time:.1f}s")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("✅ DATASET GENERATION COMPLETE!")
    print("=" * 70)
    print(f"📊 Statistics:")
    print(f"   Total examples generated: {total_generated:,}")
    print(f"   Examples per character: ~{total_generated // len(ALL_CHARACTERS):,}")
    print(f"   Total generation time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   Average speed: {total_generated/total_time:.0f} examples/second")
    print(f"\n📁 Output file: {output_path}")
    
    # Verify counts and duplicates
    print("\n🔍 Verification:")
    with open(output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        line_count = len(lines) - 1  # Subtract header
        
        # Check for duplicates
        texts = set()
        duplicate_count = 0
        reader = csv.reader(lines[1:])  # Skip header
        for row in reader:
            if len(row) > 1:
                text = row[1]
                if text in texts:
                    duplicate_count += 1
                texts.add(text)
    
    print(f"   Actual lines in CSV: {line_count:,}")
    print(f"   Unique texts: {len(texts):,}")
    print(f"   Duplicates found: {duplicate_count}")
    
    if line_count == total_generated and duplicate_count == 0:
        print("   ✓ Count verification passed - No duplicates")
    else:
        print(f"   ⚠️ Issues: expected {total_generated:,}, got {line_count:,}, duplicates: {duplicate_count}")
    
    return output_path

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Generate High-Quality IFS Character Dataset from Raw Data")
    parser.add_argument("--raw_dir", type=str, default=RAW_DATA_DIR, help="Raw data directory")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR_DEFAULT, help="Output directory")
    parser.add_argument("--examples_per_char", type=int, default=EXAMPLES_PER_CHARACTER, 
                       help=f"Examples per character (default: {EXAMPLES_PER_CHARACTER:,})")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    
    print("🎯 HIGH-QUALITY IFS CHARACTER DATASET GENERATOR")
    print("=" * 70)
    print(f"This will process raw data from: {args.raw_dir}")
    print(f"Generate {args.examples_per_char * len(ALL_CHARACTERS):,} examples")
    print(f"({args.examples_per_char:,} per character for {len(ALL_CHARACTERS)} characters)")
    print("with NO duplicates and enhanced realism from raw data.")
    print("=" * 70)
    
    # Check for raw data
    if not os.path.exists(args.raw_dir):
        print(f"\n⚠️ Warning: Raw data directory '{args.raw_dir}' does not exist.")
        print("   The system will create synthetic data using base patterns.")
        print("   For best results, add text files to the raw data directory.")
        print("   Supported formats: .txt, .csv, .json, .jsonl")
        response = input("\nContinue with synthetic data? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Generate dataset
    dataset_path = generate_high_quality_dataset(args.raw_dir, args.out_dir, args.examples_per_char)
    
    print("\n" + "=" * 70)
    print("🚀 DATASET READY FOR TRAINING!")
    print("=" * 70)
    
    # Calculate file size
    try:
        file_size = os.path.getsize(dataset_path) / 1024 / 1024
        print(f"\n📁 Dataset saved to: {dataset_path}")
        print(f"   File size: ~{file_size:.1f} MB")
        print(f"   Approx. {file_size * 1024 / line_count:.1f} KB per example")
    except:
        print(f"\n📁 Dataset saved to: {dataset_path}")
    
    print("\n💡 Training tips:")
    print("1. Use with transformers (BERT/RoBERTa) for best results")
    print("2. The 'character_cues' column can enhance model understanding")
    print("3. Consider multi-task learning with emotion detection")
    print("4. With 360,000 examples, train for 3-5 epochs")
    print("5. Use learning rate 2e-5 with linear warmup")
    print("\n🔧 Dataset features:")
    print("   • Enhanced with patterns from raw data")
    print("   • Realistic sentence structures")
    print("   • No duplicate texts")
    print("   • Perfect emotion-character matching")

if __name__ == "__main__":
    main()