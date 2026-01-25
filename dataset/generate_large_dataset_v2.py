#!/usr/bin/env python3
"""
Generate a large multilingual training dataset for NERRE.
Version 2: 100 Entity Types + 100 Relation Types
Target: 10,000+ samples across 5 languages (English, Chinese, Japanese, Korean, Thai)

🔥 ZERO-SHOT READY: ใช้ Generic Labels + Relation Aliases เพื่อให้ Model Generalize ได้ดี
🔥 V3 UPDATE: Label Synonym Augmentation - สุ่มใช้คำอธิบายแทน label ตรงๆ
"""

import json
import random
from typing import List, Dict, Any, Tuple

# ============================================================================
# 🔥 LABEL SYNONYM AUGMENTATION (Fix "Name Shifting" Problem)
# ทำให้โมเดลจำ "ความหมาย" แทนการจำ "ตัวอักษร"
# ============================================================================

# Entity Label → [Synonyms/Descriptions that mean the same thing]
ENTITY_LABEL_SYNONYMS = {
    # === Person Types ===
    "person": ["individual", "human", "someone", "a person"],
    "politician": ["political figure", "political leader", "government official", "elected official", "public servant"],
    "politicalparty": ["political party", "political organization", "political group", "party"],  # Cross-RE
    "scientist": ["researcher", "research scientist", "scientific researcher", "scholar", "academic researcher"],
    "researcher": ["scientist", "academic", "scholar", "research worker"],  # Cross-RE alias
    "musician": ["musical artist", "music performer", "singer", "instrumentalist", "recording artist"],
    "musicalartist": ["musician", "singer", "music performer", "recording artist", "musical performer"],  # Cross-RE
    "writer": ["author", "novelist", "literary writer", "book author", "storyteller"],
    "actor": ["film actor", "movie star", "performer", "thespian", "screen actor"],
    "athlete": ["sports player", "sportsperson", "professional athlete", "sports star"],
    "director": ["film director", "movie director", "filmmaker", "cinema director"],
    "artist": ["visual artist", "painter", "creative artist", "fine artist"],
    "entrepreneur": ["business founder", "startup founder", "business owner", "founder"],
    "engineer": ["technical engineer", "engineering professional", "tech engineer"],
    "doctor": ["physician", "medical doctor", "medical professional", "healthcare provider"],
    "lawyer": ["attorney", "legal professional", "legal counsel", "advocate"],
    "journalist": ["reporter", "news reporter", "media professional", "news writer"],
    "chef": ["culinary professional", "cook", "head chef", "culinary artist"],
    "celebrity": ["famous person", "public figure", "star", "well-known person"],
    "professional": ["expert", "specialist", "professional worker"],
    "author": ["writer", "book author", "novelist", "literary author"],
    
    # === Organization Types ===
    "organization": ["organisation", "institution", "group", "entity", "body"],
    "organisation": ["organization", "institution", "group", "entity"],  # Cross-RE British spelling
    "company": ["corporation", "business", "firm", "enterprise", "business company"],
    "startup": ["startup company", "new company", "emerging company", "tech startup"],
    "university": ["academic institution", "higher education institution", "college", "educational institution"],
    "government_agency": ["government body", "federal agency", "public agency", "governmental organization"],
    "nonprofit": ["non-profit organization", "charity", "charitable organization", "NGO"],
    "bank": ["financial institution", "banking company", "financial services company"],
    "hospital": ["medical center", "healthcare facility", "medical institution"],
    "research_institute": ["research center", "research organization", "research facility"],
    "institution": ["establishment", "organization", "body", "institute"],
    "band": ["music band", "musical group", "rock band", "music group"],  # Cross-RE
    
    # === Location Types ===
    "location": ["place", "area", "site", "geographical location", "locale"],
    "place": ["location", "spot", "site", "area"],
    "city": ["urban area", "municipality", "town", "metropolitan area"],
    "country": ["nation", "state", "sovereign state", "nation state"],
    "state": ["province", "region", "territory", "administrative region"],
    "region": ["area", "territory", "zone", "district"],
    "building": ["structure", "edifice", "construction", "facility"],
    "landmark": ["famous place", "notable location", "monument", "point of interest"],
    "mountain": ["peak", "mount", "mountain peak"],
    "river": ["waterway", "stream", "watercourse"],
    "island": ["isle", "land mass"],
    "continent": ["landmass", "continental region"],
    
    # === Creative Work Types ===
    "movie": ["film", "motion picture", "cinema film", "feature film"],
    "book": ["publication", "written work", "literary work", "novel"],
    "album": ["music album", "record", "studio album", "musical recording"],  # Cross-RE
    "song": ["musical track", "music track", "single", "musical piece"],  # Cross-RE
    "music_album": ["album", "record", "musical recording"],
    "tv_show": ["television show", "TV series", "television program", "TV program"],
    "game": ["video game", "computer game", "gaming title"],
    "artwork": ["art piece", "artistic work", "visual artwork"],
    "creative_work": ["artistic creation", "creative piece", "work of art"],
    "poem": ["poetry", "verse", "poetic work", "literary poem"],  # Cross-RE
    
    # === Product/Tech Types ===
    "product": ["item", "goods", "merchandise", "commercial product"],
    "software": ["computer program", "application", "software application", "program"],
    "hardware": ["computer hardware", "device", "electronic device", "equipment"],
    "app": ["mobile application", "software app", "mobile app"],
    "electronics": ["electronic device", "electronic equipment", "consumer electronics"],
    "vehicle": ["transportation", "automobile", "car", "transport vehicle"],
    "programlang": ["programming language", "coding language", "computer language"],  # Cross-RE
    "algorithm": ["computational method", "computer algorithm", "computational algorithm"],  # Cross-RE
    
    # === Science/Academic Types ===
    "academicjournal": ["academic journal", "scientific journal", "research journal", "scholarly journal"],  # Cross-RE
    "conference": ["academic conference", "scientific conference", "symposium", "meeting"],  # Cross-RE
    "discipline": ["academic discipline", "field of study", "academic field", "subject area"],  # Cross-RE
    "field": ["research field", "area of study", "domain", "specialty"],  # Cross-RE
    "theory": ["scientific theory", "theoretical framework", "hypothesis", "theoretical model"],  # Cross-RE
    "task": ["research task", "computational task", "problem", "challenge"],  # Cross-RE
    "metrics": ["measurement", "evaluation metric", "performance metric", "measure"],  # Cross-RE
    
    # === Science/Chemistry Types (Cross-RE) ===
    "chemicalcompound": ["chemical compound", "chemical substance", "molecule", "chemical"],
    "chemicalelement": ["chemical element", "element", "atomic element"],
    "protein": ["biological protein", "protein molecule", "macromolecule"],
    "enzyme": ["biological enzyme", "catalyst", "enzymatic protein"],
    "astronomicalobject": ["astronomical object", "celestial object", "space object", "celestial body"],
    
    # === Event/Time Types ===
    "event": ["occurrence", "happening", "incident", "occasion"],
    "election": ["political election", "voting", "electoral event", "vote"],  # Cross-RE
    "award": ["prize", "honor", "recognition", "accolade"],
    "competition": ["contest", "tournament", "championship"],
    "date": ["calendar date", "specific date", "day"],
    "year": ["calendar year", "specific year"],
    "month": ["calendar month"],
    "time": ["time period", "moment", "point in time"],
    "duration": ["time span", "period", "length of time"],
    "era": ["time period", "epoch", "age"],
    "temporal": ["time-related", "temporal reference"],
    
    # === Music Types (Cross-RE) ===
    "musicgenre": ["music genre", "musical genre", "style of music", "musical style"],
    "musicalinstrument": ["musical instrument", "instrument", "music instrument"],
    "literarygenre": ["literary genre", "genre of literature", "writing genre"],  # Cross-RE
    
    # === Other Types ===
    "misc": ["miscellaneous", "other", "unclassified", "general entity"],  # Cross-RE - tricky one
    "sports_team": ["athletic team", "sports club", "team", "sporting team"],
    "disease": ["medical condition", "illness", "health condition", "ailment"],
    "medicine": ["medication", "drug", "pharmaceutical", "medical treatment"],
    "food": ["edible item", "cuisine", "dish", "meal"],
    "beverage": ["drink", "liquid refreshment"],
    "language": ["spoken language", "natural language", "tongue"],
    "money": ["currency", "monetary amount", "financial amount"],
    "percent": ["percentage", "proportion", "rate"],
}

# Relation Label → [Synonyms/Descriptions]  
RELATION_LABEL_SYNONYMS = {
    # === Employment/Affiliation ===
    "works_at": ["employed at", "working for", "staff member of", "employee of", "affiliated with"],
    "role": ["has role", "serves as", "functions as", "position of"],  # Cross-RE
    "general-affiliation": ["affiliated with", "associated with", "connected to", "linked to"],  # Cross-RE
    
    # === Location Relations ===
    "located_in": ["situated in", "found in", "based in", "positioned in"],
    "org_based_in": ["headquartered in", "based in", "operates from", "main office in"],
    "lived_in": ["resided in", "living in", "dwelling in", "home in"],
    "born_in": ["birthplace", "born at", "native of", "originated from"],
    "physical": ["physically located", "physical relation", "spatial relation"],  # Cross-RE
    "origin": ["originates from", "comes from", "source of", "derived from"],  # Cross-RE
    
    # === Creation/Development ===
    "creator_of": ["created", "made", "developed", "authored", "designed"],
    "founder_of": ["founded", "established", "started", "initiated"],
    "developed": ["built", "created", "engineered", "designed"],
    "artifact": ["created artifact", "produced", "made thing"],  # Cross-RE
    
    # === Social Relations ===
    "spouse_of": ["married to", "partner of", "wedded to"],
    "social": ["social relation", "interpersonal relation", "social connection"],  # Cross-RE
    "collaborates_with": ["works with", "partners with", "cooperates with"],
    
    # === Academic/Research ===
    "graduated_from": ["studied at", "alumni of", "educated at", "attended"],
    "professor_at": ["teaches at", "faculty at", "academic at"],
    "research_at": ["researches at", "conducts research at", "studies at"],
    "topic": ["about topic", "concerning", "regarding", "subject of"],  # Cross-RE
    "usage": ["used for", "utilized for", "applied to", "employed for"],  # Cross-RE
    
    # === Performance/Entertainment ===
    "starred_in": ["appeared in", "acted in", "performed in", "featured in"],
    "plays_for": ["plays on", "member of team", "athlete for"],
    "performed_at": ["performed in", "played at", "appeared at"],
    
    # === Business Relations ===
    "acquired_by": ["bought by", "purchased by", "taken over by"],
    "subsidiary_of": ["owned by", "part of", "division of"],
    "partner_with": ["partnered with", "allied with", "in partnership with"],
    "investor_in": ["invested in", "funding", "backed"],
    
    # === Temporal Relations ===
    "founded_in": ["established in", "started in", "created in"],
    "released_in": ["launched in", "published in", "came out in"],
    "occurred_on": ["happened on", "took place on", "dated"],
    "temporal": ["time relation", "when", "during"],  # Cross-RE
    
    # === Other Relations ===
    "won": ["awarded", "received", "earned", "achieved"],
    "killed_by": ["murdered by", "slain by", "victim of"],
    "treats": ["medical treatment for", "cures", "heals"],
    "part-of": ["component of", "belongs to", "included in", "member of"],  # Cross-RE
    "type-of": ["kind of", "category of", "instance of", "subtype of"],  # Cross-RE
    "named": ["called", "known as", "titled", "named as"],  # Cross-RE
    "compare": ["compared to", "similar to", "contrasted with", "versus"],  # Cross-RE
    "cause-effect": ["causes", "results in", "leads to", "effect of"],  # Cross-RE
    "opposite": ["opposite of", "contrary to", "antonym of", "reverse of"],  # Cross-RE
    "win-defeat": ["won against", "defeated", "beat", "victory over"],  # Cross-RE
    "related-to": ["related with", "connected to", "associated with", "linked to"],  # Cross-RE
}

def get_label_synonym(label: str, label_type: str = "entity", probability: float = 0.5) -> str:
    """
    🔥 Label Synonym Augmentation
    สุ่มเปลี่ยน label เป็น synonym/description เพื่อให้โมเดลเรียนรู้ความหมาย
    
    Args:
        label: Original label (e.g., "scientist")
        label_type: "entity" or "relation"
        probability: Chance to use synonym instead of original (0.5 = 50%)
    
    Returns:
        Either original label or a random synonym
    """
    if random.random() > probability:
        return label
    
    synonyms_dict = ENTITY_LABEL_SYNONYMS if label_type == "entity" else RELATION_LABEL_SYNONYMS
    
    if label in synonyms_dict:
        synonyms = synonyms_dict[label]
        return random.choice(synonyms)
    
    return label


# ============================================================================
# CROSS-LABEL MAPPING (Hierarchical Entity Type Mapping)
# ช่วยให้ Model เรียนรู้ความสัมพันธ์ระหว่าง Label กว้างๆ กับ Label เฉพาะเจาะจง
# ============================================================================

LABEL_HIERARCHY = {
    # Location Hierarchy
    "location": ["city", "country", "state", "region", "mountain", "river", "island", "continent", "neighborhood", "landmark"],
    "place": ["location", "city", "country", "building", "airport", "stadium", "park"],
    
    # Organization Hierarchy
    "organization": ["company", "startup", "university", "government_agency", "nonprofit", "bank", "hospital", "research_institute"],
    "organization": ["company", "startup", "university", "government_agency", "nonprofit", "bank", "hospital", "research_institute"],
    "institution": ["university", "school", "hospital", "research_institute", "museum"],
    "business": ["company", "startup", "bank", "retailer", "manufacturer", "airline"],
    
    # Person Hierarchy  
    "person": ["politician", "scientist", "artist", "athlete", "musician", "actor", "director", "author", "entrepreneur", "engineer", "doctor", "lawyer", "journalist", "chef"],
    "professional": ["engineer", "doctor", "lawyer", "journalist", "chef", "scientist"],
    "celebrity": ["actor", "musician", "athlete", "artist"],
    
    # Product Hierarchy
    "product": ["software", "hardware", "vehicle", "food", "beverage", "medicine", "book", "movie", "game", "app", "electronics"],
    "creative_work": ["book", "movie", "music_album", "artwork", "tv_show", "game"],
    "tech_product": ["software", "hardware", "app", "electronics"],
    
    # Time Hierarchy
    "time": ["date", "year", "month", "duration", "era", "century", "season"],
    "temporal": ["date", "year", "month", "time", "duration"],
}

# Reverse mapping: specific → generic (for training augmentation)
LABEL_TO_PARENT = {}
for parent, children in LABEL_HIERARCHY.items():
    for child in children:
        if child not in LABEL_TO_PARENT:
            LABEL_TO_PARENT[child] = []
        LABEL_TO_PARENT[child].append(parent)

# ============================================================================
# RELATION ALIASES (Zero-Shot Ready)
# แม็พ Relations ที่มีความหมายเหมือนกันเพื่อให้ Model เข้าใจหลาย Label
# ============================================================================

RELATION_ALIASES = {
    # Employment Relations (Canonical: works_at)
    "works_at": ["employee_of", "employed_by", "works_for", "staff_of", "member_of", "hired_by", 
                 "position_at", "ceo_of", "manages", "leads", "director_of", "consultant_for"],
    
    # Organization Location (Canonical: org_based_in - CoNLL04 style)
    "org_based_in": ["headquartered_in", "based_in", "hq_in", "main_office_in", "located_in_city", "operated_in"],

    # General Location (Canonical: located_in)
    "located_in": ["situated_in", "found_in", "exists_in", "in", "part_of", "capital_of"],
    
    # Residence Relations (Canonical: lived_in - CoNLL04 style)
    "lived_in": ["lives_in", "resides_in", "dwelling_in", "home_in", "was_living_in", "resident_of"],
    
    # Violence Relations (Canonical: killed_by)
    "killed_by": ["murdered_by", "assassinated_by", "slain_by", "victim_of", "died_in"],
    
    # Birth/Origin
    "born_in": ["birthplace", "native_of", "from", "origin", "native_to"],
    
    # Education Combined (Canonical: graduated_from)
    "graduated_from": ["studied_at", "alumni_of", "attended", "educated_at", "degree_from", "enrolled_at"],

    # Creation (Canonical: creator_of - for products/works)
    "creator_of": ["made", "invented", "designed", "built", "author_of", "composed_by", "painted_by", 
                   "developer_of", "manufacturer_of", "produced_by", "inventor_of"],

    # Founding (Canonical: founder_of - for organizations)
    "founder_of": ["founded", "created", "established", "started", "co-founder_of", "co_founder_of"],
    
    # Family (Canonical: spouse_of, parent_of, child_of - Neutral/Inclusive)
    "spouse_of": ["married_to", "wife_of", "husband_of", "partner_of", "married_on"],
    "parent_of": ["father_of", "mother_of"],
    "child_of": ["son_of", "daughter_of"],
    
    # Time (Canonical: founded_in)
    "founded_in": ["established_in", "started_in", "formed_in"],
}

# สร้าง Reverse Mapping: alias → canonical
ALIAS_TO_CANONICAL = {}
for canonical, aliases in RELATION_ALIASES.items():
    for alias in aliases:
        if alias not in ALIAS_TO_CANONICAL:
            ALIAS_TO_CANONICAL[alias] = canonical
    ALIAS_TO_CANONICAL[canonical] = canonical  # map to itself too

# ============================================================================
# SENTENCE STYLE TEMPLATES (Formal, News, Narrative, Casual)
# ป้องกัน Model Overfitting บน Template เฉพาะเจาะจง
# ============================================================================

SENTENCE_STYLES = {
    "formal": {
        "en": {
            "founder_of": [
                "{person} established {company} in {date}.",
                "{person} is the founder and chairman of {company}.",
                "The establishment of {company} was undertaken by {person}.",
            ],
            "ceo_of": [
                "{person} serves as the Chief Executive Officer of {company}.",
                "{person} holds the position of CEO at {company}.",
                "The chief executive role at {company} is held by {person}.",
            ],
            "works_at": [
                "{person} is currently employed at {company}.",
                "{person} holds a position at {company}.",
                "{person} maintains employment with {company}.",
            ],
            "located_in": [
                "{company} is situated in {city}.",
                "{company} maintains its operations in {city}.",
                "The headquarters of {company} is located in {city}.",
            ],
            "graduated_from": [
                "{person} completed their education at {university}.",
                "{person} obtained their degree from {university}.",
                "{person} is an alumnus of {university}.",
            ],
        },
        "zh": {
            "founder_of": [
                "{person}於{date}正式創立了{company}。",
                "{person}是{company}的創始人兼董事長。",
            ],
            "ceo_of": [
                "{person}目前擔任{company}的首席執行官一職。",
                "{person}現任{company}執行長。",
            ],
        },
        "th": {
            "founder_of": [
                "{person} เป็นผู้ก่อตั้ง {company} เมื่อปี {date}",
                "{person} ดำรงตำแหน่งประธานผู้ก่อตั้งของ {company}",
            ],
            "ceo_of": [
                "{person} ดำรงตำแหน่งประธานเจ้าหน้าที่บริหารของ {company}",
                "{person} เป็นซีอีโอของบริษัท {company}",
            ],
        },
    },
    "news": {
        "en": {
            "founder_of": [
                "Tech mogul {person} founded {company}, according to reports.",
                "{person}, the billionaire entrepreneur, launched {company} in {date}.",
                "Breaking: {company} was founded by {person}, sources confirm.",
            ],
            "ceo_of": [
                "{person} has been named CEO of {company}, the company announced.",
                "{company} appoints {person} as new chief executive.",
                "In a major move, {person} takes the helm at {company}.",
            ],
            "works_at": [
                "{person} has joined {company}, according to insider sources.",
                "Sources say {person} is now working at {company}.",
                "{person} reportedly started at {company} this quarter.",
            ],
            "acquired_by": [
                "In a landmark deal, {company} has acquired {startup}.",
                "{startup} was purchased by {company} in a multi-billion dollar deal.",
                "Breaking: {company} completes acquisition of {startup}.",
            ],
            "won": [
                "{person} wins prestigious {award}, stunning critics.",
                "{award} goes to {person} at tonight's ceremony.",
                "In a surprise upset, {person} took home the {award}.",
            ],
        },
        "zh": {
            "founder_of": [
                "據報導，科技巨頭{person}創立了{company}。",
                "獨家：{person}於{date}成立{company}，震撼業界。",
            ],
            "acquired_by": [
                "重磅消息：{company}宣布收購{startup}。",
                "{company}以天價收購{startup}，震動市場。",
            ],
        },
        "th": {
            "founder_of": [
                "รายงาน: {person} ผู้ก่อตั้ง {company} ในปี {date}",
                "ข่าวด่วน! {person} ประกาศจัดตั้ง {company}",
            ],
            "acquired_by": [
                "ดีลใหญ่! {company} เข้าซื้อกิจการ {startup}",
                "{company} ประกาศซื้อ {startup} อย่างเป็นทางการ",
            ],
        },
    },
    "narrative": {
        "en": {
            "founder_of": [
                "It was {date} when {person} decided to start {company}.",
                "The story of {company} began with {person}'s vision.",
                "{person} had always dreamed of creating {company}.",
                "Little did anyone know that {person} would one day build {company}.",
            ],
            "ceo_of": [
                "{person} rose through the ranks to become CEO of {company}.",
                "After years of hard work, {person} finally led {company}.",
                "The journey of {person} at {company} is truly inspiring.",
            ],
            "works_at": [
                "{person} found their calling at {company}.",
                "Every day, {person} walks into {company} with a purpose.",
                "{person}'s story at {company} is just beginning.",
            ],
            "born_in": [
                "{person} grew up in {city}, dreaming of bigger things.",
                "The streets of {city} shaped {person}'s early years.",
                "{person} was born and raised in {city}.",
            ],
            "graduated_from": [
                "{person} spent their formative years studying at {university}.",
                "It was at {university} where {person} discovered their passion.",
                "After graduating from {university}, {person}'s career took off.",
            ],
        },
        "zh": {
            "founder_of": [
                "故事始於{date}，當{person}決定創立{company}時。",
                "{company}的傳奇，要從{person}的夢想說起。",
            ],
            "born_in": [
                "{person}在{city}長大，從小就有遠大的夢想。",
                "{city}是{person}的故鄉，也是他夢想開始的地方。",
            ],
        },
        "th": {
            "founder_of": [
                "เรื่องราวเริ่มต้นเมื่อ {person} ตัดสินใจก่อตั้ง {company}",
                "{person} มีความฝันที่จะสร้าง {company} มาตลอด",
            ],
            "born_in": [
                "{person} เติบโตขึ้นที่ {city} พร้อมกับความฝันอันยิ่งใหญ่",
                "ถนนทุกสายใน {city} หล่อหลอมให้ {person} เป็นอย่างทุกวันนี้",
            ],
        },
    },
    "casual": {
        "en": {
            "founder_of": [
                "So {person} basically started {company}, pretty cool right?",
                "You know {person}? Yeah, they're the one who made {company}.",
                "{person} created {company} - can you believe it?",
            ],
            "ceo_of": [
                "{person} is like the boss of {company} now.",
                "Guess who runs {company}? {person}!",
                "{person}'s the big CEO over at {company}.",
            ],
            "works_at": [
                "{person} works at {company}, you know.",
                "Did you hear? {person} got a job at {company}.",
                "{person}'s doing their thing over at {company}.",
            ],
            "lives_in": [
                "{person} lives in {city} these days.",
                "Last I heard, {person} moved to {city}.",
                "{person}'s been hanging out in {city} lately.",
            ],
            "graduated_from": [
                "{person} went to {university}, pretty impressive huh?",
                "So {person} graduated from {university}.",
                "{person}'s a {university} alum, you know.",
            ],
            "starred_in": [
                "Did you see {actor} in {movie}? So good!",
                "{actor} was in {movie}, it was amazing.",
                "I can't believe {actor} starred in {movie}!",
            ],
            "plays_for": [
                "{athlete} plays for {sports_team} now.",
                "Hey, {athlete} is on {sports_team}!",
                "{athlete}'s rocking it at {sports_team}.",
            ],
        },
        "zh": {
            "founder_of": [
                "你知道嗎？{company}是{person}創的耶！",
                "{person}搞了個{company}，厲害吧？",
            ],
            "lives_in": [
                "{person}現在住在{city}啦。",
                "聽說{person}搬到{city}了。",
            ],
            "graduated_from": [
                "{person}是{university}畢業的欸～",
                "你知道{person}讀{university}嗎？",
            ],
        },
        "th": {
            "founder_of": [
                "รู้ไหม {person} เป็นคนตั้ง {company} เอง!",
                "{person} สร้าง {company} ขึ้นมาเลยนะ เจ๋งมาก!",
            ],
            "lives_in": [
                "{person} อยู่ {city} ตอนนี้",
                "ได้ยินว่า {person} ย้ายไปอยู่ {city} แล้ว",
            ],
            "graduated_from": [
                "{person} จบจาก {university} นะเธอ",
                "เก่งมาก {person} เรียนจบจาก {university}",
            ],
            "works_at": [
                "{person} ทำงานที่ {company} นะ",
                "ได้ข่าวว่า {person} ไปทำ {company} แล้ว",
            ],
        },
        "ja": {
            "founder_of": [
                "知ってる？{company}は{person}が作ったんだよ！",
                "{person}が{company}を始めたんだって、すごいよね。",
            ],
            "works_at": [
                "{person}は{company}で働いてるみたい。",
                "聞いた？{person}が{company}に入ったって。",
            ],
        },
        "ko": {
            "founder_of": [
                "알아? {person}이 {company} 만들었대!",
                "{person}이 {company} 창업했다더라, 대박이지?",
            ],
            "works_at": [
                "{person}이 {company}에서 일해.",
                "들었어? {person}이 {company}에 입사했대.",
            ],
        },
    },
    "question": {
        "en": {
            "founder_of": [
                "Did you know that {person} founded {company}?",
                "Who founded {company}? It was {person}.",
                "Can you believe {person} started {company}?",
            ],
            "ceo_of": [
                "Who is the CEO of {company}? {person}.",
                "Did you hear that {person} became CEO of {company}?",
                "Isn't {person} the one running {company}?",
            ],
            "works_at": [
                "Does {person} still work at {company}?",
                "I wonder if {person} is at {company}.",
                "Is it true that {person} joined {company}?",
            ],
            "graduated_from": [
                "Did {person} really graduate from {university}?",
                "Where did {person} study? {university}, right?",
                "Is {person} a {university} graduate?",
            ],
        },
        "zh": {
            "founder_of": [
                "你知道{company}是{person}創立的嗎？",
                "{company}是誰創的？是{person}。",
            ],
            "ceo_of": [
                "{company}的執行長是誰？是{person}。",
                "聽說{person}當上{company}的CEO了？",
            ],
        },
        "th": {
            "founder_of": [
                "รู้ไหมว่า {person} ก่อตั้ง {company}?",
                "ใครก่อตั้ง {company}? คือ {person} นะ",
            ],
            "works_at": [
                "{person} ยังทำงานที่ {company} อยู่ไหม?",
                "จริงเหรอที่ {person} เข้าทำงานที่ {company}?",
            ],
        },
    },
}

def get_styled_template(relation_type: str, lang: str = "en", style: str = None) -> str:
    """
    ดึง Template ตาม Style ที่กำหนด หรือสุ่มถ้าไม่ระบุ
    """
    if style is None:
        style = random.choice(list(SENTENCE_STYLES.keys()))
    
    style_templates = SENTENCE_STYLES.get(style, {}).get(lang, {}).get(relation_type, [])
    
    if style_templates:
        return random.choice(style_templates), style
    
    return None, style

def apply_label_mapping_augmentation(entity_type: str, probability: float = 0.4) -> str:
    """
    สุ่มเปลี่ยน Label เฉพาะเจาะจงเป็น Label กว้างๆ (หรือกลับกัน)
    เพื่อให้ Model เรียนรู้ความสัมพันธ์ระหว่าง Label
    
    🔥 INCREASED PROBABILITY: 40% (was 30%) for better zero-shot generalization
    """
    if random.random() > probability:
        return entity_type
    
    # 60% โอกาสเปลี่ยนเป็น Parent Label (เพิ่มจาก 50%)
    if entity_type in LABEL_TO_PARENT and random.random() < 0.6:
        parent = random.choice(LABEL_TO_PARENT[entity_type])
        return parent
    
    # 40% โอกาสเปลี่ยนเป็น Sibling Label (Label อื่นในกลุ่มเดียวกัน)
    if entity_type in LABEL_TO_PARENT:
        parent = random.choice(LABEL_TO_PARENT[entity_type])
        siblings = LABEL_HIERARCHY.get(parent, [])
        if siblings:
            return random.choice(siblings)
    
    return entity_type

def canonicalize_relation_label(relation_type: str) -> str:
    """
    🔥 Strategy 1: Label Consolidation
    Collapse overlapping relations to a single canonical label to increase support.
    e.g. "ceo_of", "employed_by" -> "works_at"
    """
    if relation_type in ALIAS_TO_CANONICAL:
        return ALIAS_TO_CANONICAL[relation_type]
    
    return relation_type


# ============================================================================
# 🔥 ZERO-SHOT READY TEMPLATES - Using GENERIC Labels Directly
# ทำให้ Model เรียนรู้ "location" โดยตรง ไม่ใช่แค่ "city", "country"
# ============================================================================

GENERIC_ENTITY_TEMPLATES = [
    # === Location / Organization (CoNLL04 style) ===
    (
        "{person1} works at {organization}.",
        [("person1", "person"), ("organization", "organization")],
        [("person1", "organization", "employee_of")]  # CoNLL04 label
    ),
    (
        "{person1} was employed by {organization}.",
        [("person1", "person"), ("organization", "organization")],
        [("person1", "organization", "employee_of")]
    ),
    (
        "{organization} is based in {location}.",
        [("organization", "organization"), ("location", "location")],
        [("organization", "location", "org_based_in")]  # CoNLL04 label
    ),
    (
        "{organization} has its headquarters in {location}.",
        [("organization", "organization"), ("location", "location")],
        [("organization", "location", "org_based_in")]
    ),
    (
        "{person1} lived in {location}.",
        [("person1", "person"), ("location", "location")],
        [("person1", "location", "lived_in")]  # CoNLL04 label
    ),
    (
        "{person1} was born in {location}.",
        [("person1", "person"), ("location", "location")],
        [("person1", "location", "born_in")]
    ),
    (
        "{person1} resides in {location}.",
        [("person1", "person"), ("location", "location")],
        [("person1", "location", "lived_in")]
    ),
    # === killed_by (CoNLL04 specific) ===
    (
        "{person1} was killed by {person2}.",
        [("person1", "person"), ("person2", "person")],
        [("person1", "person2", "killed_by")]  # CoNLL04 label
    ),
    (
        "{person2} killed {person1} in the incident.",
        [("person1", "person"), ("person2", "person")],
        [("person1", "person2", "killed_by")]
    ),
    (
        "{person1} was murdered by {person2}.",
        [("person1", "person"), ("person2", "person")],
        [("person1", "person2", "killed_by")]
    ),
    (
        "{person1} was assassinated by {person2}.",
        [("person1", "person"), ("person2", "person")],
        [("person1", "person2", "killed_by")]
    ),
    (
        "{person2} shot and killed {person1}.",
        [("person1", "person"), ("person2", "person")],
        [("person1", "person2", "killed_by")]
    ),
    # === Generic Work Relations ===
    (
        "{person1} is an employee of {organization}.",
        [("person1", "person"), ("organization", "organization")],
        [("person1", "organization", "employee_of")]
    ),
    (
        "{person1} serves as a member of {organization}.",
        [("person1", "person"), ("organization", "organization")],
        [("person1", "organization", "employee_of")]
    ),
    # === Generic Location with Various Prepositions ===
    (
        "{organization} operates in {location}.",
        [("organization", "organization"), ("location", "location")],
        [("organization", "location", "org_based_in")]
    ),
    (
        "{organization} was founded in {location}.",
        [("organization", "organization"), ("location", "location")],
        [("organization", "location", "org_based_in")]
    ),
    (
        "{person1} grew up in {location}.",
        [("person1", "person"), ("location", "location")],
        [("person1", "location", "lived_in")]
    ),
    (
        "{person1} spent their childhood in {location}.",
        [("person1", "person"), ("location", "location")],
        [("person1", "location", "lived_in")]
    ),
    # === Complex Multi-Relation with Generic Labels ===
    (
        "{person1}, who works at {organization} in {location}, is a renowned expert.",
        [("person1", "person"), ("organization", "organization"), ("location", "location")],
        [("person1", "organization", "employee_of"), ("organization", "location", "org_based_in")]
    ),
    (
        "{person1} joined {organization}, which is headquartered in {location}.",
        [("person1", "person"), ("organization", "organization"), ("location", "location")],
        [("person1", "organization", "employee_of"), ("organization", "location", "org_based_in")]
    ),
    (
        "{person1} was born in {location1} but now lives in {location2}.",
        [("person1", "person"), ("location1", "location"), ("location2", "location")],
        [("person1", "location1", "born_in"), ("person1", "location2", "lived_in")]
    ),
]

# ============================================================================
# 🔥 CROSS-RE STYLE TEMPLATES (สำหรับ Zero-Shot บน Cross-RE Dataset)
# Labels: politicalparty, musicalartist, band, album, song, etc.
# Relations: role, part-of, origin, physical, topic, etc.
# ============================================================================

CROSS_RE_STYLE_TEMPLATES = [
    # === Political Domain ===
    (
        "{politician} is a member of the {politicalparty}.",
        [("politician", "politician"), ("politicalparty", "politicalparty")],
        [("politician", "politicalparty", "general-affiliation")]
    ),
    (
        "{politician} represents the {politicalparty} in parliament.",
        [("politician", "politician"), ("politicalparty", "politicalparty")],
        [("politician", "politicalparty", "role")]
    ),
    (
        "The {politicalparty} nominated {politician} as their candidate.",
        [("politicalparty", "politicalparty"), ("politician", "politician")],
        [("politician", "politicalparty", "general-affiliation")]
    ),
    (
        "{politician} left the {politicalparty} to join another party.",
        [("politician", "politician"), ("politicalparty", "politicalparty")],
        [("politician", "politicalparty", "general-affiliation")]
    ),
    (
        "{politician} won the {election} representing the {politicalparty}.",
        [("politician", "politician"), ("election", "election"), ("politicalparty", "politicalparty")],
        [("politician", "election", "win-defeat"), ("politician", "politicalparty", "general-affiliation")]
    ),
    (
        "The {election} was contested between {politician} and other candidates.",
        [("election", "election"), ("politician", "politician")],
        [("politician", "election", "role")]
    ),
    
    # === Music Domain ===
    (
        "{musicalartist} is the lead singer of {band}.",
        [("musicalartist", "musicalartist"), ("band", "band")],
        [("musicalartist", "band", "part-of")]
    ),
    (
        "{band} released their new {album} last month.",
        [("band", "band"), ("album", "album")],
        [("band", "album", "artifact")]
    ),
    (
        "The {song} is from {musicalartist}'s latest {album}.",
        [("song", "song"), ("musicalartist", "musicalartist"), ("album", "album")],
        [("song", "album", "part-of"), ("musicalartist", "album", "artifact")]
    ),
    (
        "{musicalartist} performed the {song} at the concert.",
        [("musicalartist", "musicalartist"), ("song", "song")],
        [("musicalartist", "song", "artifact")]
    ),
    (
        "{band} is known for their {musicgenre} style.",
        [("band", "band"), ("musicgenre", "musicgenre")],
        [("band", "musicgenre", "general-affiliation")]
    ),
    (
        "The {album} features songs in the {musicgenre} genre.",
        [("album", "album"), ("musicgenre", "musicgenre")],
        [("album", "musicgenre", "type-of")]
    ),
    (
        "{musicalartist} plays the {musicalinstrument} in the band.",
        [("musicalartist", "musicalartist"), ("musicalinstrument", "musicalinstrument")],
        [("musicalartist", "musicalinstrument", "usage")]
    ),
    
    # === Science/AI Domain ===
    (
        "{scientist} published a paper on {algorithm}.",
        [("scientist", "scientist"), ("algorithm", "algorithm")],
        [("scientist", "algorithm", "topic")]
    ),
    (
        "{researcher} developed the {algorithm} at {university}.",
        [("researcher", "researcher"), ("algorithm", "algorithm"), ("university", "university")],
        [("researcher", "algorithm", "artifact"), ("researcher", "university", "general-affiliation")]
    ),
    (
        "The {algorithm} was presented at {conference}.",
        [("algorithm", "algorithm"), ("conference", "conference")],
        [("algorithm", "conference", "temporal")]
    ),
    (
        "{scientist} won the {award} for their research on {field}.",
        [("scientist", "scientist"), ("award", "award"), ("field", "field")],
        [("scientist", "award", "win-defeat"), ("scientist", "field", "topic")]
    ),
    (
        "The {theory} was proposed by {scientist} in the {academicjournal}.",
        [("theory", "theory"), ("scientist", "scientist"), ("academicjournal", "academicjournal")],
        [("scientist", "theory", "artifact"), ("theory", "academicjournal", "physical")]
    ),
    (
        "{researcher} specializes in {discipline} at {university}.",
        [("researcher", "researcher"), ("discipline", "discipline"), ("university", "university")],
        [("researcher", "discipline", "topic"), ("researcher", "university", "general-affiliation")]
    ),
    (
        "The {metrics} is commonly used to evaluate {task}.",
        [("metrics", "metrics"), ("task", "task")],
        [("metrics", "task", "usage")]
    ),
    (
        "{programlang} is the preferred language for implementing {algorithm}.",
        [("programlang", "programlang"), ("algorithm", "algorithm")],
        [("programlang", "algorithm", "usage")]
    ),
    
    # === Chemistry/Biology Domain ===
    (
        "The {chemicalcompound} was discovered by {scientist}.",
        [("chemicalcompound", "chemicalcompound"), ("scientist", "scientist")],
        [("scientist", "chemicalcompound", "artifact")]
    ),
    (
        "{protein} interacts with {enzyme} in the cell.",
        [("protein", "protein"), ("enzyme", "enzyme")],
        [("protein", "enzyme", "physical")]
    ),
    (
        "{chemicalelement} is a key component of {chemicalcompound}.",
        [("chemicalelement", "chemicalelement"), ("chemicalcompound", "chemicalcompound")],
        [("chemicalelement", "chemicalcompound", "part-of")]
    ),
    
    # === Literature Domain ===
    (
        "{writer} wrote the {book} in the {literarygenre} style.",
        [("writer", "writer"), ("book", "book"), ("literarygenre", "literarygenre")],
        [("writer", "book", "artifact"), ("book", "literarygenre", "type-of")]
    ),
    (
        "The {poem} by {writer} won the {award}.",
        [("poem", "poem"), ("writer", "writer"), ("award", "award")],
        [("writer", "poem", "artifact"), ("poem", "award", "win-defeat")]
    ),
    
    # === Astronomy Domain ===
    (
        "{scientist} discovered the {astronomicalobject}.",
        [("scientist", "scientist"), ("astronomicalobject", "astronomicalobject")],
        [("scientist", "astronomicalobject", "artifact")]
    ),
    (
        "The {astronomicalobject} is located in the {location} constellation.",
        [("astronomicalobject", "astronomicalobject"), ("location", "location")],
        [("astronomicalobject", "location", "physical")]
    ),
    
    # === Cross-RE Specific Relations ===
    (
        "{person} is also known as {misc}.",
        [("person", "person"), ("misc", "misc")],
        [("person", "misc", "named")]
    ),
    (
        "The {algorithm} is similar to {algorithm2} but more efficient.",
        [("algorithm", "algorithm"), ("algorithm2", "algorithm")],
        [("algorithm", "algorithm2", "compare")]
    ),
    (
        "{event} led to the formation of {organisation}.",
        [("event", "event"), ("organisation", "organisation")],
        [("event", "organisation", "cause-effect")]
    ),
    (
        "Unlike {method1}, {method2} uses a different approach.",
        [("method1", "algorithm"), ("method2", "algorithm")],
        [("method1", "method2", "opposite")]
    ),
    (
        "The {concept1} is related to {concept2} in several ways.",
        [("concept1", "misc"), ("concept2", "misc")],
        [("concept1", "concept2", "related-to")]
    ),
    (
        "{person1} collaborated with {person2} on the project.",
        [("person1", "person"), ("person2", "person")],
        [("person1", "person2", "social")]
    ),
    (
        "The conference was held in {location}, {country}.",
        [("location", "location"), ("country", "country")],
        [("location", "country", "physical")]
    ),
    (
        "{organisation} originated from {country}.",
        [("organisation", "organisation"), ("country", "country")],
        [("organisation", "country", "origin")]
    ),
]

# === CONLL04-STYLE TEMPLATES (Specific to that dataset's relations) ===
CONLL04_STYLE_TEMPLATES = [
    # Employee_Of variations
    (
        "{person1}, a longtime employee of {organization}, announced their retirement.",
        [("person1", "person"), ("organization", "organization")],
        [("person1", "organization", "employee_of")]
    ),
    (
        "According to sources, {person1} has been working at {organization} for many years.",
        [("person1", "person"), ("organization", "organization")],
        [("person1", "organization", "employee_of")]
    ),
    (
        "{person1} started their career at {organization}.",
        [("person1", "person"), ("organization", "organization")],
        [("person1", "organization", "works_at")]  # Also train on works_at
    ),
    # Org_Based_In variations
    (
        "The {organization}, based in {location}, reported strong earnings.",
        [("organization", "organization"), ("location", "location")],
        [("organization", "location", "org_based_in")]
    ),
    (
        "{location} is home to {organization}, a major employer in the region.",
        [("organization", "organization"), ("location", "location")],
        [("organization", "location", "org_based_in")]
    ),
    # Lived_In variations  
    (
        "{person1}, a resident of {location}, was interviewed yesterday.",
        [("person1", "person"), ("location", "location")],
        [("person1", "location", "lived_in")]
    ),
    (
        "During the 1990s, {person1} lived in {location}.",
        [("person1", "person"), ("location", "location")],
        [("person1", "location", "lived_in")]
    ),
    (
        "{person1} moved to {location} in search of better opportunities.",
        [("person1", "person"), ("location", "location")],
        [("person1", "location", "lived_in")]
    ),
    # Killed_By variations (news style)
    (
        "{person1} was fatally shot by {person2} yesterday.",
        [("person1", "person"), ("person2", "person")],
        [("person1", "person2", "killed_by")]
    ),
    (
        "In a tragic turn of events, {person2} killed {person1}.",
        [("person1", "person"), ("person2", "person")],
        [("person1", "person2", "killed_by")]
    ),
    (
        "{person1} died after being attacked by {person2}.",
        [("person1", "person"), ("person2", "person")],
        [("person1", "person2", "killed_by")]
    ),
    (
        "Police confirmed that {person1} was murdered by {person2}.",
        [("person1", "person"), ("person2", "person")],
        [("person1", "person2", "killed_by")]
    ),
]

# ============================================================================
# 🔥 LINGUISTIC PARAPHRASE TEMPLATES
# หนีจาก Template! ใช้ Passive Voice, Appositive, Relative Clause
# ============================================================================

LINGUISTIC_PARAPHRASE_TEMPLATES = [
    # =====================================================================
    # PASSIVE VOICE - ประธานถูกกระทำ
    # =====================================================================
    # works_at / employee_of (Passive)
    (
        "{organization} employed {person1} as a senior executive.",
        [("organization", "organization"), ("person1", "person")],
        [("person1", "organization", "works_at")]
    ),
    (
        "{person1} was hired by {organization} last year.",
        [("person1", "person"), ("organization", "organization")],
        [("person1", "organization", "works_at")]
    ),
    (
        "A position at {organization} was offered to {person1}.",
        [("organization", "organization"), ("person1", "person")],
        [("person1", "organization", "works_at")]
    ),
    
    # located_in / org_based_in (Passive)
    (
        "{location} is where {organization} has established its headquarters.",
        [("location", "location"), ("organization", "organization")],
        [("organization", "location", "located_in")]
    ),
    (
        "The main office of {organization} was set up in {location}.",
        [("organization", "organization"), ("location", "location")],
        [("organization", "location", "located_in")]
    ),
    
    # lived_in (Passive)
    (
        "{location} was where {person1} spent most of their life.",
        [("location", "location"), ("person1", "person")],
        [("person1", "location", "lived_in")]
    ),
    (
        "{person1} was raised in {location} before moving abroad.",
        [("person1", "person"), ("location", "location")],
        [("person1", "location", "lived_in")]
    ),
    
    # killed_by (Passive - already passive, add variations)
    (
        "{person1} was shot dead by {person2} during the conflict.",
        [("person1", "person"), ("person2", "person")],
        [("person1", "person2", "killed_by")]
    ),
    (
        "The death of {person1} was caused by {person2}.",
        [("person1", "person"), ("person2", "person")],
        [("person1", "person2", "killed_by")]
    ),
    
    # =====================================================================
    # APPOSITIVE - คำขยายความหลังคอมม่า
    # =====================================================================
    # works_at with Appositive
    (
        "{person1}, an engineer, works at {organization}.",
        [("person1", "person"), ("organization", "organization")],
        [("person1", "organization", "works_at")]
    ),
    (
        "{person1}, a senior manager, is employed by {organization}.",
        [("person1", "person"), ("organization", "organization")],
        [("person1", "organization", "works_at")]
    ),
    (
        "{organization}, a leading tech company, employs {person1}.",
        [("organization", "organization"), ("person1", "person")],
        [("person1", "organization", "works_at")]
    ),
    
    # located_in with Appositive
    (
        "{organization}, a multinational corporation, is headquartered in {location}.",
        [("organization", "organization"), ("location", "location")],
        [("organization", "location", "located_in")]
    ),
    (
        "{location}, a major business hub, hosts the headquarters of {organization}.",
        [("location", "location"), ("organization", "organization")],
        [("organization", "location", "located_in")]
    ),
    
    # lived_in with Appositive
    (
        "{person1}, a former resident, lived in {location} for decades.",
        [("person1", "person"), ("location", "location")],
        [("person1", "location", "lived_in")]
    ),
    (
        "{location}, a coastal city, was home to {person1}.",
        [("location", "location"), ("person1", "person")],
        [("person1", "location", "lived_in")]
    ),
    
    # killed_by with Appositive
    (
        "{person1}, a prominent figure, was killed by {person2}.",
        [("person1", "person"), ("person2", "person")],
        [("person1", "person2", "killed_by")]
    ),
    (
        "{person2}, a suspect, allegedly killed {person1}.",
        [("person2", "person"), ("person1", "person")],
        [("person1", "person2", "killed_by")]
    ),
    
    # =====================================================================
    # RELATIVE CLAUSE - ประโยคย่อยขยายความ (who, which, that, where)
    # =====================================================================
    # works_at with Relative Clause
    (
        "{person1}, who joined recently, works at {organization}.",
        [("person1", "person"), ("organization", "organization")],
        [("person1", "organization", "works_at")]
    ),
    (
        "{organization}, which was founded in 1990, employs {person1}.",
        [("organization", "organization"), ("person1", "person")],
        [("person1", "organization", "works_at")]
    ),
    (
        "The company that hired {person1} is {organization}.",
        [("person1", "person"), ("organization", "organization")],
        [("person1", "organization", "works_at")]
    ),
    (
        "{person1} is employed by {organization}, which is known for innovation.",
        [("person1", "person"), ("organization", "organization")],
        [("person1", "organization", "works_at")]
    ),
    
    # located_in with Relative Clause
    (
        "{organization}, which operates globally, is based in {location}.",
        [("organization", "organization"), ("location", "location")],
        [("organization", "location", "located_in")]
    ),
    (
        "{location}, where many companies are located, hosts {organization}.",
        [("location", "location"), ("organization", "organization")],
        [("organization", "location", "located_in")]
    ),
    (
        "The city where {organization} has its headquarters is {location}.",
        [("organization", "organization"), ("location", "location")],
        [("organization", "location", "located_in")]
    ),
    
    # lived_in with Relative Clause
    (
        "{person1}, who grew up there, lived in {location}.",
        [("person1", "person"), ("location", "location")],
        [("person1", "location", "lived_in")]
    ),
    (
        "{location}, where {person1} resided, is a beautiful place.",
        [("location", "location"), ("person1", "person")],
        [("person1", "location", "lived_in")]
    ),
    (
        "The city that {person1} called home was {location}.",
        [("person1", "person"), ("location", "location")],
        [("person1", "location", "lived_in")]
    ),
    
    # killed_by with Relative Clause
    (
        "{person1}, who was unarmed, was killed by {person2}.",
        [("person1", "person"), ("person2", "person")],
        [("person1", "person2", "killed_by")]
    ),
    (
        "{person2}, who fled the scene, killed {person1}.",
        [("person2", "person"), ("person1", "person")],
        [("person1", "person2", "killed_by")]
    ),
    (
        "The person that killed {person1} was {person2}.",
        [("person1", "person"), ("person2", "person")],
        [("person1", "person2", "killed_by")]
    ),
    
    # =====================================================================
    # COMPLEX COMBINATIONS - Passive + Appositive + Relative
    # =====================================================================
    (
        "{person1}, a distinguished scientist who studied at Oxford, works at {organization}, which is headquartered in {location}.",
        [("person1", "person"), ("organization", "organization"), ("location", "location")],
        [("person1", "organization", "works_at"), ("organization", "location", "located_in")]
    ),
    (
        "{organization}, a company that was founded in {location}, has employed {person1} since 2010.",
        [("organization", "organization"), ("location", "location"), ("person1", "person")],
        [("organization", "location", "located_in"), ("person1", "organization", "works_at")]
    ),
    (
        "{person1}, who previously lived in {location1}, now resides in {location2} and works at {organization}.",
        [("person1", "person"), ("location1", "location"), ("location2", "location"), ("organization", "organization")],
        [("person1", "location1", "lived_in"), ("person1", "location2", "lived_in"), ("person1", "organization", "works_at")]
    ),
    (
        "In {location}, where {organization} is based, {person1} was hired as the new director.",
        [("location", "location"), ("organization", "organization"), ("person1", "person")],
        [("organization", "location", "located_in"), ("person1", "organization", "works_at")]
    ),
    (
        "{person1}, an employee of {organization}, lived in {location} before relocating.",
        [("person1", "person"), ("organization", "organization"), ("location", "location")],
        [("person1", "organization", "works_at"), ("person1", "location", "lived_in")]
    ),
    
    # =====================================================================
    # INVERTED WORD ORDER - สลับลำดับคำ
    # =====================================================================
    (
        "Based in {location} is {organization}, a Fortune 500 company.",
        [("location", "location"), ("organization", "organization")],
        [("organization", "location", "located_in")]
    ),
    (
        "Working at {organization} is {person1}, a top executive.",
        [("organization", "organization"), ("person1", "person")],
        [("person1", "organization", "works_at")]
    ),
    (
        "Living in {location} is {person1}, a retired professor.",
        [("location", "location"), ("person1", "person")],
        [("person1", "location", "lived_in")]
    ),
    (
        "At {organization}, {person1} serves as the chief scientist.",
        [("organization", "organization"), ("person1", "person")],
        [("person1", "organization", "works_at")]
    ),
    (
        "In {location}, {organization} maintains its global headquarters.",
        [("location", "location"), ("organization", "organization")],
        [("organization", "location", "located_in")]
    ),
]

# ============================================================================
# 100 ENTITY TYPES - Organized by Category
# ============================================================================

ENTITY_TYPES = {
    # === PERSON (15 types) ===
    "person": "An individual human being, person, or citizen referred to in a general context.",
    "politician": "A political leader, government official, head of state, minister, or person active in party politics and public office.",
    "scientist": "A researcher, academic professor, scientist, or scholar engaged in systematic study, discovery, and scientific experimentation.",
    "artist": "A creative individual, visual artist, painter, sculptor, or person who produces art through various mediums.",
    "athlete": "A professional sports player, athlete, competitor, or person who participates in organized physical sports and matches.",
    "musician": "A musical performer, singer, composer, conductor, or member of a music band or orchestra.",
    "actor": "A performer in films, television shows, theater, or stage plays, including actors and actresses.",
    "director": "A person who supervises the creative production of movies, films, television series, or theatrical plays.",
    "author": "A writer of books, novels, poetry, or articles; an individual who creates written literary works.",
    "entrepreneur": "A business founder, startup creator, investor, or individual who establishes and manages new commercial ventures.",
    "engineer": "A technical expert in software, hardware, civil, or industrial engineering who designs and builds complex systems.",
    "doctor": "A medical professional, physician, surgeon, or healthcare specialist licensed to practice medicine and treat patients.",
    "lawyer": "A legal professional, attorney, counselor, or solicitor who provides legal advice and represents clients in court.",
    "journalist": "A news reporter, correspondent, journalist, or anchor who gathers and broadcasts news through media outlets.",
    "chef": "A professional cook, executive chef, or culinary expert specializing in food preparation and kitchen management.",
    
    # === ORGANIZATION (20 types) ===
    "organization": "A general group of people, association, or organized body with a particular purpose, not fitting other specific categories.",

    "company": "A business corporation, commercial firm, enterprise, or profit-seeking entity providing goods or services.",
    "startup": "A young, newly established entrepreneurial venture or high-growth company, often in the early stages of development or technology focus.",
    "nonprofit": "A non-profit organization (NGO), charity, foundation, or voluntary association operating for social or public benefit rather than profit.",
    "government_agency": "A state department, public authority, government agency, or administrative body of a city, state, or nation.",
    "university": "A higher education institution, college, university, or academic body focused on tertiary teaching and advanced research.",
    "school": "An educational institution for primary or secondary students, including high schools, elementary schools, and vocational colleges.",
    "hospital": "A medical facility, hospital, health center, or clinic where patients receive professional treatment and healthcare services.",
    "bank": "A financial institution, bank, investment firm, or credit union that handles money, loans, and financial transactions.",
    "airline": "An aviation company, airline, or air carrier that provides transport services for passengers or freight by aircraft.",
    "sports_team": "A professional or amateur sports team, athletic club, or franchise participating in organized sports competitions.",
    "military": "The armed forces, military organization, army, navy, or air force of a country concerned with national defense.",
    "political_party": "A political organization, party, or alliance that seeks to influence government policy and participate in elections.",
    "media_company": "A news outlet, broadcasting company, publisher, media house, or organization involved in television, radio, or digital news.",
    "research_institute": "A laboratory, scientific center, or research institute dedicated to systematic investigation and technological innovation.",
    "museum": "A public institution, art gallery, or museum that cares for and displays a collection of artifacts, art, or historical objects.",
    "restaurant": "A food service establishment, restaurant, cafe, or dining chain where meals are prepared and served to customers.",
    "hotel": "A lodging establishment, hotel, resort, or inn that provides accommodation, rooms, and hospitality services.",
    "retailer": "A retail store, e-commerce shop, shopping center, or merchant selling goods directly to consumers.",
    "manufacturer": "A manufacturing company, industrial plant, or factory engaged in the large-scale production of goods and hardware.",
    
    # === LOCATION (15 types) ===
    "location": "A general geographic place, site, or physical point on Earth not specified by other categories.",
    "city": "A large human settlement, urban area, city, town, or municipality with a specific name.",
    "country": "A sovereign nation, country, or independent state with its own territory and government.",
    "state": "A major administrative subdivision of a country, such as a state, province, or prefecture.",
    "continent": "One of the world's major continuous expanses of land, such as Asia, Europe, or Africa.",
    "region": "A broad geographic area, district, or territory with common characteristics, often spanning multiple cities.",
    "building": "A physical human-made structure with a roof and walls, such as a house, office, or factory.",
    "landmark": "A recognizable natural or man-made feature, famous monument, or historic site used for navigation or tourism.",
    "airport": "A complex for air transportation, airport, airfield, or terminal where aircraft take off and land.",
    "stadium": "A large sports arena, stadium, or venue with seating for spectators used for events and matches.",
    "park": "A public green space, national park, nature reserve, or area of land kept for recreation.",
    "island": "A piece of land entirely surrounded by water, which is smaller than a continent.",
    "mountain": "A large natural elevation of the Earth's surface, mountain peak, or range rising abruptly from the surrounding level.",
    "river": "A natural flowing watercourse, river, stream, or large body of water like a lake or ocean.",
    "neighborhood": "A residential district, neighborhood, or specific community area within a larger city or town.",

    
    # === TIME (8 types) ===
    "date": "A specific calendar day, including day, month, and year, or a particular day of the week.",
    "year": "A specific four-digit calendar year or a reference to a particular year in time.",
    "month": "One of the twelve months of the year, such as January, February, or a specific monthly period.",
    "time": "A specific point in time during the day, including hours, minutes, and exact clock time.",
    "duration": "An amount or period of time that something lasts, such as hours, days, or weeks of elapsed time.",
    "era": "A long and distinct period of history, a historical era, an age, or a specific geological epoch.",
    "century": "A period of one hundred years, typically referred to as a specific century like the 21st century.",
    "season": "One of the four divisions of the year (spring, summer, autumn, winter) or a specific recurring period.",

    
    # === PRODUCT (15 types) ===
    "product": "A general physical or digital item, commodity, or manufactured good offered for sale or use.",
    "software": "A computer program, software application, suite of tools, or system software used on computing devices.",
    "hardware": "Physical computing equipment, computer hardware, internal components, or mechanical devices.",
    "vehicle": "A mode of transport, vehicle, car, truck, aircraft, or boat used for moving people or goods.",
    "food": "An edible substance, food product, dish, ingredient, or culinary item consumed for nutrition.",
    "beverage": "A drinkable liquid, beverage, soft drink, or alcoholic drink intended for human consumption.",
    "medicine": "A pharmaceutical drug, medicine, medication, vaccine, or therapeutic substance used for medical treatment.",
    "book": "A written literary work, book, novel, textbook, or printed publication consisting of pages.",
    "movie": "A cinematic film, movie, motion picture, or video production shown in theaters or on screens.",
    "music_album": "A collection of audio recordings, music album, record, or specific song released by an artist.",
    "game": "An interactive video game, electronic game, or tabletop game played for entertainment.",
    "app": "A mobile application, web app, or specific software tool designed for smartphones or tablets.",
    "electronics": "A consumer electronic device, gadget, appliance, or hardware powered by electricity.",
    "clothing": "An item of apparel, clothing, fashion garment, or wearable accessory for the human body.",
    "cosmetics": "A beauty product, cosmetic, skincare item, makeup, or personal care substance used for grooming.",


    
    # === TECHNOLOGY (10 types) ===
    "programlang": "A formal programming language, syntax, or computer coding language used to write software and instructions.",
    "framework": "A software framework, library, or set of pre-written code and tools used as a platform for developing applications.",
    "database": "A database management system (DBMS), structured data storage, or organized collection of digital information.",
    "protocol": "A network communication protocol, set of rules for data exchange, or digital communication standard.",
    "api": "An Application Programming Interface (API), web service endpoint, or interface for software-to-software interaction.",
    "algorithm": "A mathematical procedure, computer algorithm, data structure, or step-by-step logic for solving a specific task.",
    "os": "An operating system (OS) that manages computer hardware, software resources, and provides common services for programs.",
    "ai_model": "An artificial intelligence model, machine learning algorithm, neural network, or trained AI system architecture.",
    "cryptocurrency": "A digital currency, cryptocurrency token, decentralized digital asset, or blockchain-based financial instrument.",
    "technology": "A general scientific or technical innovation, method, or specialized system not covered by other technology categories.",


    
    # === EVENT (8 types) ===
    "event": "A general occurrence, happening, or organized social activity that takes place at a specific time and location.",
    "conference": "A formal meeting, professional summit, academic symposium, or organized gathering for discussion and exchange of information.",
    "festival": "A public celebration, cultural festival, holiday event, or organized series of performances and festivities.",
    "war": "A state of armed conflict, war, military battle, or prolonged period of fighting between nations or groups.",
    "election": "A formal process of democratic voting, political election, or public referendum to choose a political leader or representative.",
    "disaster": "A sudden accident, natural disaster (like an earthquake or flood), or man-made catastrophe causing widespread damage.",
    "ceremony": "A formal religious or public occasion, award ceremony, gala, or ritual event marking a particular opening or achievement.",
    "competition": "A sports tournament, contest, match, or competitive event where individuals or teams strive for victory and prizes.",
    
    # === CREATIVE WORKS (5 types) ===
    "artwork": "A unique creative work of visual art, painting, sculpture, sketch, or artistic installation created by an artist.",
    "patent": "A legal document for intellectual property, a patent, or a registered invention grant that protects an innovator's rights.",
    "invention": "A new and unique device, method, composition, or process that has been created or discovered through innovation.",
    "research_paper": "An academic paper, scientific publication, journal article, or scholarly report describing original research findings.",
    "tv_show": "A television series, broadcast show, episodic program, or serial production created for TV or streaming platforms.",
    
    # === CONCEPTS (4 types) ===
    "award": "A formal prize, honor, medal, or recognition bestowed for achievement, excellence, or merit.",
    "degree": "An academic qualification, degree, diploma, or certificate conferred by an educational institution upon completion of study.",
    "title": "A professional job title, official position, rank, or designated role held by an individual within an organization.",
    "skill": "A specific technical skill, expertise, area of knowledge, or professional competence that a person possesses.",




    # === NUMERICAL & FINANCIAL (4 types) ===
    "money": "Specific monetary values, amounts of currency, financial wealth, or net worth expressed in units like dollars, baht, or euros.",
    "percent": "A numerical value expressed as a fraction of one hundred, including interest rates, tax rates, or percentage shares of ownership.",
    "quantity": "A generic numerical count, amount, or number of units, items, or physical objects mentioned in the text.",
    "stock_symbol": "A unique series of letters or numbers representing a particular publicly traded company on a stock exchange, such as a ticker symbol.",


    "url": "A web address, uniform resource locator (URL), website link, or digital URI pointing to a specific page on the internet.",
    "email": "An electronic mail address (email) used for digital communication, typically containing the '@' symbol and a domain name.",
    "phone_number": "A telecommunication number, telephone, or mobile contact number, including international country codes and local area codes.",
    "ip_address": "A unique network identifier or internet protocol (IP) address, including both IPv4 and IPv6 numerical formats.",


    "legal_document": "A formal legal document, law, legislative act, constitution, statute, or specific section of a legal code or contract.",
    "language": "A natural human language, spoken dialect, or specific tongue used for communication between people of different nations.",
    "academic_field": "A specific branch of knowledge, subject of study, academic discipline, or major field of research and education.",


    "disease": "A specific medical condition, illness, sickness, disorder, or physical symptom affecting the health of an individual.",
    "medicine": "A pharmaceutical drug, chemical compound, medication, or vaccine used to treat, prevent, or cure medical diseases.",
    "organ": "A specific body part, internal organ, or biological system within a living organism, such as the heart, lung, or immune system.",
    
    # === NATURE (2 types) ===
    "animal": "A living organism belonging to the kingdom Animalia, including mammals, birds, reptiles, and fish.",
    "plant": "A living organism belonging to the kingdom Plantae, including trees, flowers, herbs, and shrubs.",
    
    # === OBJECTS (1 type) ===
    "instrument": "A device created or adapted to make musical sounds, played by a musician.",


}

# ============================================================================
# 100 RELATION TYPES - Organized by Category
# ============================================================================

RELATION_TYPES = {
    # === CREATION/OWNERSHIP (15 types) ===
    "founder_of": ("person", "organization", "Established or started a new company, institution, or group from the beginning"),
    "ceo_of": ("person", "organization", "Is the highest-ranking executive officer in a business or corporate organization"),
    "owner_of": ("person", "organization", "Legally possesses or holds the title of ownership over an entity or property"),
    "creator_of": ("person", "product", "The person responsible for the artistic or intellectual creation of a creative work"),
    "inventor_of": ("person", "invention", "The person who conceived and designed a new, unique technological device or process"),
    "author_of": ("person", "book", "The person who wrote the original literary content of a book, novel, or document"),
    "director_of": ("person", "movie", "The person who oversaw the creative aspects and directed the actors in a movie or film"),
    "producer_of": ("person", "product", "The person or group responsible for the financial and administrative management of a creative product"),
    "designed_by": ("product", "person", "The aesthetic or functional plan of a product was created by this specific person"),
    "developed": ("organization", "product", "A company or organization that engineered and brought a new product or software to market"),
    "manufactured_by": ("product", "organization", "The physical production or assembly of a product was done by this industrial company"),
    "published_by": ("book", "organization", "The publishing house or company that printed and distributed a book or written work"),
    "composed_by": ("music_album", "musician", "Composed/written by this specific musician or composer"),
    "painted_by": ("artwork", "artist", "The artwork was created/painted by this specific artist"),
    "patented_by": ("patent", "person", "A specific individual who was legally granted a patent for an invention by a government authority"),
    
    # === LOCATION (12 types) ===
    "located_in": ("organization", "location", "An organization or physical entity is geographically situated within a specific location"),
    "headquartered_in": ("company", "city", "A business company has its primary administrative office in this specific city"),
    "born_in": ("person", "location", "A human being was physically born at this geographic location or country"),
    "died_in": ("person", "location", "A human being passed away or died at this specific geographic location"),
    "lives_in": ("person", "city", "A person currently resides or has their home in this specific city or town"),
    "operates_in": ("company", "country", "A commercial company conducts its business activities within this specific country"),
    "based_in": ("organization", "city", "The main operations or headquarters of an organization are situated in this city"),
    "filmed_in": ("movie", "location", "The production of a movie took place at this specific geographic location"),
    "held_in": ("event", "location", "An organized event or occurrence took place at this specific location"),
    "native_to": ("person", "country", "A person originally comes from or has ancestral roots in this specific country"),
    "capital_of": ("city", "country", "This city serves as the official seat of government for a sovereign country"),
    "part_of": ("location", "location", "A smaller geographic area is contained within or belongs to a larger region or territory"),
    
    # === TIME (10 types) ===
    "founded_in": ("organization", "date", "A business or organization was officially established in this specific calendar year"),
    "released_in": ("product", "date", "A product was made available to the public in this specific calendar year"),
    "born_on": ("person", "date", "A human being's date of birth is this specific calendar day"),
    "died_on": ("person", "date", "A human being's date of death is this specific calendar day"),
    "started_in": ("event", "date", "An organized event or occurrence began in this specific calendar year"),
    "ended_in": ("event", "date", "An organized event or occurrence concluded in this specific calendar year"),
    "established_in": ("organization", "date", "A business or organization was officially established in this specific calendar year"),
    "occurred_on": ("event", "date", "An organized event or occurrence took place on this specific calendar day"),
    "graduated_in": ("person", "date", "A person completed their academic degree or education in this specific calendar year"),
    "married_on": ("person", "date", "Two individuals were legally wed on this specific calendar day"),
    
    # === EMPLOYMENT (10 types) ===
    "works_at": ("person", "organization", "Is employed by or works for a specific organization or company"),
    "employed_by": ("person", "company", "Is hired and receives a salary from a specific business or corporate entity"),
    "position_at": ("person", "organization", "Holds a specific job title or role within an organization or company"),
    "manages": ("person", "organization", "Oversees and is responsible for the operations of a specific organization or team"),
    "leads": ("person", "organization", "Directs and guides a specific organization, department, or group of people"),
    "reports_to": ("person", "person", "Is subordinate to and takes instructions from another individual in a hierarchy"),
    "hired_by": ("person", "company", "Was recruited and brought on board by a specific business or corporate entity"),
    "resigned_from": ("person", "company", "Voluntarily left or quit a position at a specific organization or company"),
    "retired_from": ("person", "organization", "Ended their professional career at a specific organization or company"),
    "consultant_for": ("person", "company", "Provides expert advice and services to a specific business or corporate entity"),
    
    # === EDUCATION (8 types) ===
    "studied_at": ("person", "university", "Attended as a student at a specific educational institution"),
    "graduated_from": ("person", "university", "Completed a degree or academic program at a specific educational institution"),
    "degree_from": ("person", "university", "Earned an academic degree from a specific educational institution"),
    "professor_at": ("scientist", "university", "Holds a teaching or research position at a specific university or college"),
    "teaches_at": ("person", "school", "Is an instructor or educator at a specific school or educational institution"),
    "research_at": ("scientist", "research_institute", "Conducts scientific research at a specific research institute or laboratory"),
    "alumni_of": ("person", "university", "Is a former student or graduate of a specific educational institution"),
    "dropout_from": ("person", "university", "Left or discontinued studies at a specific educational institution before graduation"),
    
    # === FAMILY (20 types) ===
    "spouse_of": ("person", "person", "Legally married partner of another individual"),
    "parent_of": ("person", "person", "Biological or adoptive mother or father of a child"),
    "child_of": ("person", "person", "Biological or adopted son or daughter of a parent"),
    "sibling_of": ("person", "person", "Brother or sister sharing at least one parent"),
    "relative_of": ("person", "person", "A family member or kinship relation between two individuals"),
    "married_to": ("person", "person", "Legally wedded to another individual"),
    "divorced_from": ("person", "person", "Legally ended marriage with another individual"),
    "partner_of": ("person", "person", "In a romantic or domestic partnership with another individual"),
    "father_of": ("person", "person", "Biological or adoptive male parent of a child"),
    "mother_of": ("person", "person", "Biological or adoptive female parent of a child"),
    "son_of": ("person", "person", "Biological or adopted male child of a parent"),
    "daughter_of": ("person", "person", "Biological or adopted female child of a parent"),
    "brother_of": ("person", "person", "Biological or adopted   male sibling"),
    "sister_of": ("person", "person", "Biological or adopted female sibling"),
    "grandparent_of": ("person", "person", "Biological or adoptive grandmother or grandfather of a grandchild"),
    "grandchild_of": ("person", "person", "Biological or adopted grandson or granddaughter of a grandparent"),
    "aunt_of": ("person", "person", "Biological or adoptive sister of a parent"),
    "uncle_of": ("person", "person", "Biological or adoptive brother of a parent"),
    "cousin_of": ("person", "person", "Child of an aunt or uncle"),
    "niece_of": ("person", "person", "Daughter of a sibling"),
    "nephew_of": ("person", "person", "Son of a sibling"),


    
    # === BUSINESS (35 types) ===
    "subsidiary_of": ("company", "company", "A company that is completely or partly owned and controlled by another company"),
    "acquired_by": ("company", "company", "Was purchased or taken over by another company"),
    "merged_with": ("company", "company", "Combined with another company to form a single entity"),
    "partner_with": ("company", "company", "Has a formal business partnership or collaboration with another company"),
    "competitor_of": ("company", "company", "Operates in the same market and offers similar products or services as another company"),
    "investor_in": ("person", "company", "Provided capital or funding to a specific business or startup"),
    "invested_by": ("startup", "company", "Received investment or funding from a specific business or investor"),
    "supplies_to": ("company", "company", "Provides goods or services to another company as a supplier"),
    "customer_of": ("company", "company", "Purchases products or services from another company as a client"),
    "distributor_of": ("company", "product", "Acts as a middleman to sell and distribute products for another company"),
    "licensed_by": ("product", "company", "Was granted legal permission to use or sell a product by a specific company"),
    "sponsored_by": ("event", "company", "Received financial support or sponsorship from a specific business or organization"),
    "endorsed_by": ("product", "person", "Was publicly supported or promoted by a specific individual or celebrity"),
    "franchise_of": ("company", "company", "Operates under the brand and business model of a larger parent company"),
    "head_coach_of": ("person", "sports_team", "Is the main coach responsible for training and leading a sports team"),
    "team_captain_of": ("person", "sports_team", "Is the designated leader and representative of a sports team"),
    "signed_by": ("athlete", "sports_team", "Was officially contracted to play for a specific sports team"),
    "endorses_product": ("person", "product", "Publicly supports or promotes a specific product or brand"),
    "sells": ("retailer", "product", "Offers specific products for sale to consumers"),
    "distributes": ("company", "product", "Handles the logistics and distribution of products to retailers or customers"),
    "manufactures": ("manufacturer", "product", "Produces and assembles specific products in a factory or industrial setting"),
    "exports_to": ("company", "country", "Sends goods or services to another country for sale or trade"),
    "imports_from": ("company", "country", "Brings in goods or services from another country for domestic use"),
    "listed_on": ("company", "stock_symbol", "Is publicly traded on a specific stock exchange under a ticker symbol"),
    "traded_on": ("stock_symbol", "bank", "Is bought and sold on a specific stock exchange or financial market"),
    "headquartered_at": ("organization", "address", "The main office location of an organization is situated at this specific address"),
    "leads":("person","person","Is the leader of a specific group or organization"),
    "founded":("person","organization","Established or started a new company, institution, or group from the beginning"),
    "owns":("person","organization","Legally possesses or holds the title of ownership over an entity or property"),
    "created":("person","product","The person responsible for the artistic or intellectual creation of a creative work"),
    "subordinate_of":("person","person","Is under the authority or control of another individual in a hierarchy"),
    "employs":("organization","person","Hires and pays a person to work for the organization"),
    "trains":("person","person","Provides instruction and skill development to another individual"),
    "mentors":("person","person","Offers guidance and advice to a less experienced individual"),
    "collaborates_with":("organization","organization","Works jointly with another organization on a project or initiative"),

    # === ASSOCIATION (20 types) ===
    "member_of": ("person", "organization", "Member of"),
    "affiliated_with": ("person", "organization", "Affiliated with"),
    "belongs_to": ("product", "company", "Belongs to"),
    "represents": ("person", "country", "Represents"),
    "ambassador_for": ("person", "organization", "Ambassador for"),
    "spokesperson_for": ("person", "company", "Spokesperson for"),
    "endorses": ("person", "product", "Endorses"),
    "supports": ("person", "political_party", "Supports"),
    "advocates_for": ("person", "cause", "Advocates for"),
    "donated_to": ("person", "nonprofit", "Donated to"),
    "volunteers_at": ("person", "nonprofit", "Volunteers at"),
    "member_of_team": ("athlete", "sports_team", "Member of team"),
    "represented_by": ("athlete", "agent", "Represented by"),
    "sponsored": ("athlete", "company", "Sponsored by"),
    "trained_by": ("athlete", "coach", "Trained by"),
    "competes_in": ("athlete", "competition", "Competes in"),
    "holds_membership_in": ("person", "organization", "Holds membership in"),
    "certified_by": ("person", "organization", "Certified by"),
    "accredited_by": ("organization", "organization", "Accredited by"),
    "licensed_to": ("person", "organization", "Licensed to"),

    
    # === AWARDS & ACHIEVEMENTS (7 types) ===
    "won": ("person", "award", "Won award"),
    "nominated_for": ("person", "award", "Nominated for"),
    "recipient_of": ("person", "award", "Recipient of"),
    "awarded_by": ("award", "organization", "Awarded by"),
    "achieved": ("person", "title", "A person earned or was granted a formal honorific title or status"),
    "holds_record": ("person", "event", "Holds record in"),
    "champion_of": ("athlete", "competition", "Champion of"),
    

    # === MEDIA & ENTERTAINMENT (20 types) ===
    "starred_in": ("actor", "movie", "Starred in"),
    "appeared_in": ("person", "tv_show", "Appeared in"),
    "performed_at": ("musician", "event", "Performed at"),
    "interviewed_by": ("person", "journalist", "Interviewed by"),
    "featured_in": ("person", "media_company", "Featured in"),
    "hosts": ("person", "tv_show", "Hosts"),
    "plays_for": ("athlete", "sports_team", "Plays for"),
    "coached_by": ("athlete", "person", "Coached by"),
    "signed_with": ("athlete", "sports_team", "Signed with"),
    "transferred_to": ("athlete", "sports_team", "Transferred to"),
    "directed_by": ("movie", "director", "Directed by"),
    "produced_by": ("movie", "producer", "Produced by"),
    "written_by": ("movie", "author", "Written by"),
    "composed_for": ("musician", "movie", "Composed music for"),
    "published_by": ("book", "publisher", "Published by"),
    "adapted_from": ("movie", "book", "Adapted from"),
    "based_on": ("movie", "real_event", "Based on a real event or story"),
    "remake_of": ("movie", "movie", "Remake of an earlier film"),
    "sequel_to": ("movie", "movie", "Sequel to a previous film"),
    "prequel_to": ("movie", "movie", "Prequel to a later film"),
    

    # === FINANCIAL (30 types) ===
    "has_net_worth": ("person", "money", "Has a total estimated net worth of"),
    "valuation_of": ("money", "company", "The market valuation of a company"),
    "invested_amount": ("person", "money", "Amount invested by a person"),
    "holds_shares_of": ("person", "percent", "Percentage of shares held in a company"),
    "market_cap": ("company", "money", "Market capitalization of a company"),
    "sold_for": ("product", "money", "Product or company sold for this amount"),
    "revenue_of": ("company", "money", "Annual or period revenue of a company"),
    "salary_of": ("person", "money", "Estimated salary or compensation of a person"),
    "funded_amount": ("company", "money", "Amount of funding raised by a company"),
    "acquisition_cost": ("company", "money", "Cost of acquiring another company"),
    "profit_of": ("company", "money", "Net profit or earnings of a company"),
    "loss_of": ("company", "money", "Financial loss incurred by a company"),
    "dividend_yield": ("company", "percent", "Dividend yield percentage of a company"),
    "interest_rate_of": ("bank", "percent", "Interest rate offered by a bank"),
    "loan_amount": ("person", "money", "Amount of loan taken by a person"),
    "credit_score_of": ("person", "number", "Credit score of a person"),
    "budget_of": ("organization", "money", "Allocated budget for a project or department"),
    "expenditure_of": ("organization", "money", "Total expenditure or spending of an organization"),
    "tax_rate_of": ("company", "percent", "Corporate tax rate applicable to a company"),
    "financial_aid_of": ("person", "money", "Amount of financial aid received by a person"),
    "sponsorship_amount": ("event", "money", "Amount of sponsorship funding for an event"),
    "royalty_rate": ("product", "percent", "Royalty rate percentage for a licensed product"),
    "subscription_fee": ("service", "money", "Fee charged for a subscription service"),
    "transaction_amount": ("person", "money", "Amount involved in a financial transaction"),
    "asset_value": ("company", "money", "Total value of assets owned by a company"),
    "liability_amount": ("company", "money", "Total liabilities owed by a company"),
    "equity_percentage": ("person", "percent", "Percentage of equity ownership held by a person"),
    "diversified_portfolio": ("person", "investment", "Has a diversified investment portfolio"),
    "financial_institution_of": ("person", "bank", "Primary financial institution used by a person"),
    "credit_limit_of": ("person", "money", "Credit limit assigned to a person by a financial institution"),


    # === DIGITAL & MARKET (20 types) ===
    "official_website": ("organization", "url", "The official URL of an organization"),
    "listed_as": ("company", "stock_symbol", "Company is listed under this ticker symbol"),
    "download_url": ("software", "url", "The download link for a software/app"),
    "social_media": ("person", "url", "Social media profile link of a person"),
    "trading_on": ("stock_symbol", "bank", "Stock symbol traded on a specific exchange"),
    "customer_support_email": ("company", "email", "Customer support email address of a company"),
    "contact_number": ("organization", "phone_number", "Contact phone number of an organization"),
    "ip_registered_to": ("ip_address", "organization", "The organization to which an IP address is registered"),
    "website_hosted_by": ("url", "organization", "The hosting provider of a website"),
    "app_available_on": ("app", "platform", "The platform where an app is available for download"),
    "uses_protocol": ("software", "protocol", "The network protocol used by a software application"),
    "built_with_framework": ("software", "framework", "The software framework used to build an application"),
    "powered_by_ai_model": ("software", "ai_model", "The AI model that powers a software application"),
    "stores_data_in": ("software", "database", "The database system used to store data for a software application"),
    "supports_language": ("software", "programlang", "The programming language supported by a software application"),
    "runs_on_os": ("software", "os", "The operating system on which a software application runs"),
    "accepts_payment_in": ("ecommerce_platform", "cryptocurrency", "The cryptocurrency accepted as payment by an e-commerce platform"),
    "uses_algorithm": ("software", "algorithm", "The specific algorithm implemented in a software application"),
    "offers_subscription_at": ("service", "money", "The subscription fee charged by a digital service"),
    "provides_api": ("software", "api", "The API offered by a software application for integration"),


    # === HEALTHCARE (20 types) ===
    "treats": ("medicine", "disease", "A pharmaceutical substance or clinical therapy used by doctors to cure a specific illness"),
    "diagnosed_with": ("person", "disease", "Person diagnosed with a medical condition"),
    "affects": ("disease", "organ", "Disease that affects specific body parts"),
    "dosage_of": ("medicine", "quantity", "Recommended dosage amount"),
    "developed_vaccine": ("company", "medicine", "Company developed a specific vaccine"),
    "approved_by": ("medicine", "regulatory_agency", "Medicine approved by a health authority"),
    "side_effects_of": ("medicine", "symptom", "Adverse side effects caused by a medicine"),
    "prescribed_for": ("medicine", "disease", "Medicine prescribed to treat a specific illness"),
    "clinical_trial_conducted_by": ("medicine", "research_institute", "Clinical trial conducted by a research institute"),
    "symptom_of": ("disease", "symptom", "A specific symptom associated with a disease"),
    "transmitted_by": ("disease", "vector", "Disease transmitted through a specific vector"),
    "prevented_by": ("disease", "medicine", "Disease that can be prevented by a specific medicine or vaccine"),
    "cured_by": ("disease", "medicine", "Disease that can be cured by a specific medicine or treatment"),
    "diagnosed_at": ("person", "hospital", "Person diagnosed at a specific medical facility"),
    "treated_at": ("person", "hospital", "Person treated at a specific medical facility"),
    "research_on": ("scientist", "disease", "Scientist conducting research on a specific disease"),
    "genetic_marker_for": ("gene", "disease", "A specific gene associated with a disease"),
    "vaccine_for": ("medicine", "disease", "A vaccine developed to protect against a specific disease"),
    "approved_for_use_by": ("medicine", "regulatory_agency", "Medicine approved for use by a health authority"),
    "manufactured_at": ("medicine", "pharmaceutical_company", "Medicine manufactured at a specific pharmaceutical company"),



    "governs": ("organization", "location", "Has official authority over a geographic area"),
    "head_of_state": ("person", "country", "A person serving as the formal leader of a sovereign nation, such as a monarch or president"),
    "member_of_parliament": ("person", "organization", "An elected representative serving in a legislative body"),
    "allied_with": ("country", "country", "Has a formal alliance or partnership with another country"),
    "sanctioned_by": ("person", "organization", "Was penalized or restricted by a governing body"),
    "ratified": ("organization", "legal_document", "Formally approved or confirmed a legal document or treaty"),
    "vetoed_by": ("legal_document", "person", "Was rejected or blocked by a specific individual with veto power"),
    "enforced_by": ("legal_document", "organization", "Is implemented and upheld by a specific governing body"),

    "scientific_discovery": ("scientist", "invention", "Made a significant scientific breakthrough or invention"),
    "published_in": ("research_paper", "journal", "Was published in a specific academic journal"),
    "cited_by": ("research_paper", "research_paper", "Referenced in another research paper"),
    "collaborated_with": ("scientist", "scientist", "Worked jointly with another scientist on research"),
    "funded_by": ("research_at", "organization", "Received financial support from an organization"),
    "hypothesis_of": ("theory", "scientist", "Proposed by a specific scientist"),
    "clinical_trial_of": ("medicine", "disease", "Tested in clinical trials for a specific disease"),
    "sequenced": ("scientist", "gene", "Determined the order of nucleotides in a specific gene"),
    "peer_reviewed_by": ("research_paper", "scientist", "Evaluated by experts in the field before publication"),
    "experimental_data_from": ("research_paper", "research_institute", "Data collected from experiments conducted at a research institute"),


    "launched_by": ("satellite", "organization", "Sent into space by a specific space agency or company"),
    "orbits": ("celestial_object", "celestial_object", "Revolves around another celestial body"),
    "landed_on": ("spacecraft", "celestial_object", "Successfully touched down on the surface of a celestial body"),
    "observed_by": ("celestial_object", "research_institute", "Studied or monitored by a specific research institute"),
    "mission_of": ("spacecraft", "organization", "A specific space mission undertaken by an organization"),
    "reusable_launch_vehicle": ("rocket", "organization", "A rocket designed for multiple launches by a specific organization"),


    "plaintiff_in": ("person", "legal_case", "The party bringing a lawsuit"),
    "defendant_in": ("person", "legal_case", "The party being sued or accused in a legal case"),
    "presided_over_by": ("legal_case", "judge", "The judge who oversaw a legal case"),
    "convicted_of": ("person", "crime", "Found guilty of a specific criminal offense"),
    "settled_with": ("company", "company", "Resolved a legal dispute with another company"),
    "infringes_on": ("product", "patent", "Violates the intellectual property rights of a patented invention"),
    "compliant_with": ("organization", "legal_document", "Adheres to the regulations outlined in a legal document"),

    "exhibited_at": ("artwork", "museum", "Displayed at a specific museum or gallery"),
    "discovered_at": ("archaeological_site", "location", "Found at a specific geographic location"),
    "excavated_by": ("archaeological_site", "scientist", "Unearthed or dug up by a specific archaeologist or team"),
    "dated_to": ("artifact", "date", "Estimated to originate from a specific historical period or calendar year"),
    "restored_by": ("artwork", "organization", "Repaired or preserved by a specific organization"),
    "historical_figure_in": ("person", "event", "A person who played a significant role in a historical event"),
    "influenced_by": ("artist", "person", "Artistic style or work was influenced by another individual"),
    "dedicated_to": ("monument", "person", "A monument or structure is dedicated in honor of a specific individual"),

    # === NEW RELATIONS ===
    "speaks": ("person", "language", "The person speaks, writes, or communicates in this language"),
    "plays": ("musician", "instrument", "The musician plays this musical instrument"),
    "found_in": ("animal", "location", "The animal is naturally found or lives in this location"),
    "eats": ("animal", "food", "The animal consumes this food"),

}

# ============================================================================
# Entity Database - Real-world entities for each type and language
# ============================================================================

# === ENGLISH ENTITIES ===
EN_ENTITIES = {
    "person": [
        "Elon Musk", "Bill Gates", "Steve Jobs", "Jeff Bezos", "Mark Zuckerberg",
        "Tim Cook", "Satya Nadella", "Sundar Pichai", "Larry Page", "Sergey Brin",
        "Warren Buffett", "Sam Altman", "Jensen Huang", "Lisa Su", "Pat Gelsinger",
        "James Smith", "Maria Garcia", "Robert Johnson", "Lisa Miller", "Michael Davis",
        "Jennifer Rodriguez", "William Martinez", "Linda Hernandez", "David Wilson", "Elizabeth Anderson",
        "Richard Thomas", "Barbara Taylor", "Joseph Moore", "Susan Jackson", "Thomas Martin",
        "Margaret Lee", "Charles Perez", "Jessica Thompson", "Christopher White", "Sarah Harris",
        "Daniel Sanchez", "Karen Clark", "Matthew Ramirez", "Nancy Lewis", "Anthony Robinson",
        "Lisa Walker", "Mark Young", "Betty Hall", "Donald Allen", "Dorothy King",
        "Steven Wright", "Sandra Scott", "Paul Torres", "Ashley Nguyen", "Andrew Hill",
        "Kimberly Flores", "Joshua Green", "Donna Adams", "Kenneth Nelson", "Emily Baker",
        "Kevin Hall", "Michelle Rivera", "Brian Campbell", "Carol Mitchell", "George Carter",
        "Amanda Roberts", "Edward Gomez", "Melissa Phillips", "Ronald Evans", "Deborah Turner"
    ],
    "politician": [
        "Barack Obama", "Donald Trump", "Joe Biden", "Angela Merkel", "Emmanuel Macron",
        "Justin Trudeau", "Boris Johnson", "Vladimir Putin", "Xi Jinping", "Narendra Modi",
        "Kamala Harris", "Jacinda Ardern", "Sanna Marin", "Pedro Sánchez", "Cyril Ramaphosa",
        "Ursula von der Leyen", "Giorgia Meloni", "Yoshihide Suga", "Scott Morrison", "Imran Khan",
        "Recep Tayyip Erdoğan", "Mohammed bin Salman", "Sheikh Hasina", "Alexander De Croo", "Mette Frederiksen"
        
    ],
    "scientist": [
        "Albert Einstein", "Stephen Hawking", "Marie Curie", "Isaac Newton", "Charles Darwin",
        "Nikola Tesla", "Richard Feynman", "Neil deGrasse Tyson", "Michio Kaku", "Jane Goodall",
        "Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio", "Fei-Fei Li", "Demis Hassabis",
        "Jennifer Doudna", "Emmanuelle Charpentier", "Katalin Karikó", "Frances Arnold", "Sara Seager",
        "Tim Berners-Lee", "Ada Lovelace", "Rosalind Franklin", "Carl Sagan", "James Watson"
    ],
    "artist": [
        "Leonardo da Vinci", "Pablo Picasso", "Vincent van Gogh", "Claude Monet", "Andy Warhol",
        "Salvador Dalí", "Frida Kahlo", "Banksy", "Yayoi Kusama", "Ai Weiwei",
        "Georgia O'Keeffe", "Jackson Pollock", "Henri Matisse", "Edvard Munch", "Gustav Klimt",
        "Rembrandt", "Michelangelo", "Raphael", "Caravaggio", "Paul Cézanne"
    ],
    "athlete": [
        "Michael Jordan", "LeBron James", "Cristiano Ronaldo", "Lionel Messi", "Serena Williams",
        "Tiger Woods", "Roger Federer", "Usain Bolt", "Muhammad Ali", "Tom Brady",
        "Naomi Osaka", "Lewis Hamilton", "Michael Phelps", "Simone Biles", "Kobe Bryant",
        "Stephen Curry", "Rafael Nadal", "Novak Djokovic", "Megan Rapinoe", "Kevin Durant",
        "Virat Kohli", "Kylian Mbappé", "Eliud Kipchoge", "Caeleb Dressel", "Sydney McLaughlin",
        "Sifan Hassan", "Allyson Felix", "Katie Ledecky", "Gabby Douglas", "Shaun White",
        "Carli Lloyd", "James Harden", "Anthony Joshua", "Canelo Álvarez", "Conor McGregor", "Simone Manuel"
    ],
    "musician": [
        "Taylor Swift", "Beyoncé", "Ed Sheeran", "Drake", "The Weeknd",
        "BTS", "Ariana Grande", "Bruno Mars", "Lady Gaga", "Rihanna",
        "Adele", "Justin Bieber", "Billie Eilish", "Coldplay", "Dua Lipa",
        "Kendrick Lamar", "Post Malone", "Olivia Rodrigo", "Shawn Mendes", "Harry Styles",
        "Lizzo", "Doja Cat", "Sam Smith", "Halsey", "Miley Cyrus"
    ],
    "actor": [
        "Leonardo DiCaprio", "Tom Hanks", "Meryl Streep", "Robert Downey Jr.", "Scarlett Johansson",
        "Dwayne Johnson", "Jennifer Lawrence", "Brad Pitt", "Angelina Jolie", "Chris Hemsworth",
        "Keanu Reeves", "Will Smith", "Emma Watson", "Timothée Chalamet", "Zendaya",
        "Gal Gadot", "Chris Evans", "Margot Robbie", "Ryan Reynolds", "Natalie Portman",
        "Samuel L. Jackson", "Charlize Theron", "Hugh Jackman", "Amy Adams", "Daniel Radcliffe"
    ],
    "director": [
        "Steven Spielberg", "Christopher Nolan", "Martin Scorsese", "Quentin Tarantino", "James Cameron",
        "Denis Villeneuve", "Greta Gerwig", "Bong Joon-ho", "Ridley Scott", "Peter Jackson",
        "Clint Eastwood", "Guillermo del Toro", "Sofia Coppola", "Wes Anderson", "David Fincher",
        "Alfonso Cuarón", "Taika Waititi", "Jordan Peele", "Kathryn Bigelow", "Spike Lee",
        "James Wan", "Ron Howard", "Ang Lee", "Sam Mendes", "Joel Coen"
    ],
    "author": [
        "J.K. Rowling", "Stephen King", "George R.R. Martin", "Dan Brown", "Haruki Murakami",
        "Margaret Atwood", "Neil Gaiman", "Yuval Noah Harari", "Malcolm Gladwell", "James Clear",
        "Isabel Allende", "Chimamanda Ngozi Adichie", "Paulo Coelho", "John Grisham", "Sally Rooney",
        "Colson Whitehead", "Brandon Sanderson", "V.E. Schwab", "Celeste Ng", "Tara Westover",
        "Anthony Doerr", "Zadie Smith", "Khaled Hosseini", "Gillian Flynn", "Donna Tartt"
    ],
    "entrepreneur": [
        "Richard Branson", "Jack Ma", "Larry Ellison", "Michael Bloomberg", "Oprah Winfrey",
        "Marc Benioff", "Reid Hoffman", "Peter Thiel", "Travis Kalanick", "Brian Chesky",
        "Whitney Wolfe Herd", "Evan Spiegel", "Drew Houston", "Ben Silbermann", "Stewart Butterfield",
        "Jessica Alba", "Sara Blakely", "Daymond John", "Kevin Systrom", "Jan Koum",
        "Elizabeth Holmes", "Fred Smith", "Howard Schultz", "Indra Nooyi", "Sheryl Sandberg"
    ],
    "engineer": [
        "Linus Torvalds", "Guido van Rossum", "Brendan Eich", "James Gosling", "Dennis Ritchie",
        "Ken Thompson", "Bjarne Stroustrup", "Anders Hejlsberg", "John Carmack", "Margaret Hamilton",
        "Radia Perlman", "Tim Sweeney", "Grace Hopper", "Ada Lovelace", "Hedy Lamarr",
        "Vint Cerf", "Bob Kahn", "Steve Wozniak", "Donald Knuth", "Alan Turing",
        "Claude Shannon", "John von Neumann", "Elon Musk", "Ginni Rometty", "Satya Nadella"
    ],
    "doctor": [
        "Anthony Fauci", "Sanjay Gupta", "Oz Mehmet", "Ben Carson", "Atul Gawande",
        "Paul Farmer", "Leana Wen", "Vishal Rao", "Rochelle Walensky", "David Sinclair",
        "Jennifer Doudna", "Emmanuelle Charpentier", "Katalin Karikó", "Siddhartha Mukherjee", "Peter Hotez",
        "Harold Varmus", "Francis Collins", "Eric Topol", "Zubin Damania", "Gail Cassell",
        "Susan Love", "Otis Brawley", "Robert Wachter", "Catherine DeAngelis", "Paul Offit"
    ],
    "journalist": [
        "Anderson Cooper", "Christiane Amanpour", "Wolf Blitzer", "Rachel Maddow", "Tucker Carlson",
        "Lester Holt", "Gayle King", "Jake Tapper", "Don Lemon", "Martha Raddatz",
        "Fareed Zakaria", "Megyn Kelly", "Chris Wallace", "Bret Baier", "Judy Woodruff",
        "David Muir", "Norah O'Donnell", "Scott Pelley", "George Stephanopoulos", "Bill O'Reilly", "Charlie Rose",
        "Katie Couric", "Tom Brokaw", "Diane Sawyer", "Barbara Walters", "Dan Rather"
    ],
    "chef": [
        "Gordon Ramsay", "Jamie Oliver", "Anthony Bourdain", "Wolfgang Puck", "Massimo Bottura",
        "Alice Waters", "Thomas Keller", "Emeril Lagasse", "Rachael Ray", "Bobby Flay",
        "Ina Garten", "Nigella Lawson", "Heston Blumenthal", "David Chang", "José Andrés",
        "Curtis Stone", "Giada De Laurentiis", "Alain Ducasse", "Marco Pierre White", "Paul Bocuse",
        "Yotam Ottolenghi", "Grant Achatz", "Dominique Crenn", "Daniel Boulud", "Rick Bayless"
    ],
    
    # Organizations
    "organization": [
        "United Nations", "World Health Organization", "Red Cross", "Amnesty International", "Greenpeace",
        "Doctors Without Borders", "World Wildlife Fund", "UNICEF", "Oxfam", "Habitat for Humanity",
        "Save the Children", "CARE International", "Human Rights Watch", "The Nature Conservancy", "Mercy Corps",
        "International Rescue Committee", "Plan International", "Action Against Hunger", "Global Witness", "Transparency International",
        "World Food Programme", "International Monetary Fund", "World Bank", "Interpol", "WTO"
    ],
    "company": [
        "Apple", "Microsoft", "Google", "Amazon", "Meta", "Tesla", "NVIDIA", "Intel", "AMD", "IBM",
        "Oracle", "Salesforce", "Adobe", "Netflix", "Spotify", "Uber", "Airbnb", "Twitter", "LinkedIn",
        "Snapchat", "Pinterest", "Dropbox", "Slack", "Zoom", "Shopify", "eBay", "PayPal", "Square", "Snap Inc.", "Reddit", "TikTok",
        "ByteDance", "Huawei", "Samsung", "Sony", "LG", "Dell", "HP", "Cisco", "Qualcomm", "Siemens", "Accenture",
        "SAP", "Infosys", "Tata Consultancy Services", "Capgemini", "Cognizant", "Wipro", "HCL Technologies"
    ],
    "startup": [
        "OpenAI", "Anthropic", "Stripe", "Databricks", "Canva", "Figma", "Notion", "Airtable", "Vercel",
        "Snowflake", "UiPath", "Robinhood", "Coinbase", "Palantir", "SpaceX", "Rivian", "Nuro", "Lime", "ChargePoint",
        "Cameo", "Chime", "Brex", "Ginkgo Bioworks", "Impossible Foods", "Beyond Meat", "DoorDash", "Instacart", "Postmates",
        "Coupang", "Grab", "Gojek", "Ola Cabs"
    ],
    "university": [
        "Harvard University", "Stanford University", "MIT", "Oxford University", "Cambridge University",
        "Yale University", "Princeton University", "Columbia University", "UC Berkeley", "Caltech",
        "University of Chicago", "UCLA", "University of Toronto", "ETH Zurich", "University of Tokyo",
        "National University of Singapore", "Tsinghua University", "Peking University", "University of Melbourne", "University of Edinburgh",
        "University of British Columbia", "University of Michigan", "Cornell University", "Duke University", "Johns Hopkins University"
    ],
    "sports_team": [
        "Los Angeles Lakers", "New York Yankees", "Real Madrid", "Barcelona FC", "Manchester United",
        "Golden State Warriors", "Dallas Cowboys", "New England Patriots", "Chicago Bulls",
        "Boston Red Sox", "Liverpool FC", "Bayern Munich", "Paris Saint-Germain", "Toronto Raptors", "Miami Heat",
        "San Francisco 49ers", "Seattle Seahawks", "Cleveland Cavaliers", "Houston Rockets", "Arsenal FC",
        "Juventus", "AC Milan", "Chelsea FC", "Manchester City", "Atletico Madrid",
    ],
    "bank": [
        "JPMorgan Chase", "Bank of America", "Goldman Sachs", "Morgan Stanley", "Citibank",
        "Wells Fargo", "HSBC", "Deutsche Bank", "Credit Suisse", "Barclays",
        "UBS", "BNP Paribas", "Royal Bank of Canada", "TD Bank", "Santander",
        "ING Group", "Societe Generale", "Mizuho Financial Group", "Sumitomo Mitsui Banking Corporation", "Commonwealth Bank",
        "ANZ", "Westpac", "Scotiabank", "Rabobank", "Nordea"
    ],
    "airline": [
        "United Airlines", "Delta Airlines", "American Airlines", "Emirates", "Singapore Airlines",
        "Lufthansa", "British Airways", "Qatar Airways", "Air France", "Southwest Airlines",
        "Cathay Pacific", "ANA", "KLM", "Turkish Airlines", "Etihad Airways",
        "Qantas", "Air Canada", "Japan Airlines", "Iberia", "Alaska Airlines",
        "Virgin Atlantic", "Aeroflot", "Saudia", "Ethiopian Airlines", "LATAM Airlines"
    ],
    "media_company": [
        "CNN", "BBC", "The New York Times", "The Washington Post", "Reuters",
        "Bloomberg", "Fox News", "MSNBC", "The Guardian", "Wall Street Journal",
        "NBC News", "CBS News", "Al Jazeera", "The Economist", "Financial Times",
        "Vox Media", "BuzzFeed", "HuffPost", "Vice Media", "The Atlantic", "Politico",
        "Axios", "CNET", "TechCrunch", "Wired", "The Verge"
    ],
    "research_institute": [
        "NASA", "CERN", "NIH", "Max Planck Institute", "MIT Media Lab",
        "DeepMind", "OpenAI Research", "Google Brain", "FAIR", "Microsoft Research",
        "Bell Labs", "Salk Institute", "Broad Institute", "Cold Spring Harbor Laboratory", "Tsinghua University Research Institute",  
        "Fraunhofer Society", "Los Alamos National Laboratory", "Lawrence Berkeley National Laboratory", "Argonne National Laboratory", "Oak Ridge National Laboratory",
        "Riken Institute", "Karolinska Institute", "Weizmann Institute of Science", "Institute Pasteur", "Johns Hopkins Applied Physics Laboratory"
    ],
    "hospital": [
        "Mayo Clinic", "Cleveland Clinic", "Johns Hopkins Hospital", "Massachusetts General Hospital",
        "UCLA Medical Center", "Mount Sinai Hospital", "Toronto General Hospital", "Charité – Universitätsmedizin Berlin",
        "Singapore General Hospital", "Royal Melbourne Hospital", "Karolinska University Hospital", "St Thomas' Hospital",
        "Cedars-Sinai Medical Center", "NYU Langone Health", "Houston Methodist Hospital",
        "Duke University Hospital", "Vancouver General Hospital", "Sheba Medical Center", "Apollo Hospitals", "Asan Medical Center"
    ],
    "manufacturer": [
        "Samsung Electronics", "Foxconn", "TSMC", "Qualcomm", "Broadcom", "Texas Instruments",
        "Sony Corporation", "LG Electronics", "Panasonic", "Hitachi", "Siemens", "GE Appliances", "Whirlpool",
        "Bosch", "Mitsubishi Electric", "Sharp Corporation", "Toshiba", "Lenovo", "Acer", "ASUS", "Dell Technologies",
        "HP Inc.", "Fujitsu", "Canon Inc.", "Nikon Corporation", "GoPro", "DJI", "Garmin", "Fitbit", "Sonos", "JBL",
        "Bose", "Yamaha", "Sennheiser", "Harman Kardon", "Vizio", "ZTE", "Oppo", "Vivo", "OnePlus"
    ],
    "retailer": [
        "Walmart", "Amazon", "Costco", "Target", "Home Depot", "Best Buy", "IKEA",
        "Lowe's", "Kroger", "Aldi", "Tesco", "Carrefour", "Metro AG", "JD.com", "Alibaba",
        "eBay", "Rakuten", "Flipkart", "Macy's", "Nordstrom", "Sears", "Dillard's", "Kohl's", "Wayfair", "Zara", "H&M",
        "Uniqlo", "Gap Inc.", "Old Navy", "Forever 21", "Urban Outfitters", "ASOS"
    ],
    
    # Locations
    "location": [
        "Silicon Valley", "Wall Street", "Hollywood", "Times Square", "Central Park",
        "Golden Gate Bridge", "Grand Canyon", "Mount Everest", "Sahara Desert", "Great Barrier Reef",
        "Niagara Falls", "Yellowstone National Park", "Yosemite National Park", "Statue of Liberty", "Mount Fuji",
        "Eiffel Tower", "Colosseum", "Big Ben", "Sydney Opera House", "Christ the Redeemer",
        "Pyramids of Giza", "Machu Picchu", "Angkor Wat", "Stonehenge", "Petra",
        "Mount Kilimanjaro", "Galápagos Islands", "Serengeti National Park", "Banff National Park", "Lake Tahoe"
    ],
    "city": [
        "San Francisco", "New York", "Los Angeles", "Seattle", "Boston", "Chicago", "Austin",
        "London", "Paris", "Tokyo", "Singapore", "Hong Kong", "Shanghai", "Beijing", "Seoul",
        "Sydney", "Toronto", "Berlin", "Amsterdam", "Dubai", "Mumbai", "Bangalore",
        "Mexico City", "São Paulo", "Buenos Aires", "Cape Town", "Cairo", "Moscow",
        "Istanbul", "Rome", "Madrid", "Lisbon", "Vienna", "Prague", "Dublin", "Edinburgh", "Vancouver", "Melbourne"
    ],
    "country": [
        "United States", "China", "Japan", "Germany", "United Kingdom", "France", "India",
        "Canada", "Australia", "South Korea", "Brazil", "Italy", "Spain", "Russia",
        "Mexico", "Indonesia", "Netherlands", "Saudi Arabia", "Turkey", "Switzerland",
        "Sweden", "Norway", "Argentina", "South Africa", "Egypt", "Thailand", "Vietnam", "Philippines", "New Zealand", "Poland",
        "Belgium", "Austria", "Denmark", "Finland", "Ireland", "Greece", "Czech Republic", "Hungary",
        "Portugal", "Chile", "Colombia", "Peru", "Ukraine", "Romania", "Slovakia", "Bulgaria",
        "Croatia", "Slovenia", "Luxembourg", "Iceland", "Estonia", "Latvia", "Lithuania", "Malta"
    ],
    "state": [
        "California", "Texas", "New York", "Florida", "Washington", "Massachusetts", "Colorado",
        "Illinois", "Ohio", "Pennsylvania", "Virginia", "Michigan", "New Jersey", "North Carolina",
        "Georgia", "Tennessee", "Arizona", "Oregon", "Minnesota", "Wisconsin", "Indiana",
        "South Carolina", "Alabama", "Kentucky", "Louisiana", "Mississippi", "Iowa", "Nebraska", "Kansas",
        "Oklahoma", "Arkansas", "Utah", "Nevada", "New Mexico", "Hawaii", "Maine", "Vermont", "New Hampshire", "Rhode Island", "Delaware",
        "West Virginia", "Montana", "Idaho", "Wyoming", "North Dakota", "South Dakota", "Alaska"
    ],
    "building": [
        "Empire State Building", "Burj Khalifa", "One World Trade Center", "Taipei 101",
        "Shanghai Tower", "Petronas Towers", "Willis Tower", "The Shard", "CN Tower", "Lotte World Tower",
        "Marina Bay Sands", "Jin Mao Tower", "International Commerce Centre", "Kingdom Centre", "Abraj Al Bait",
        "Canton Tower", "Federation Tower", "432 Park Avenue", "30 St Mary Axe", "Bank of America Tower", "Comcast Center",
        "John Hancock Center", "Chrysler Building", "Flatiron Building", "Walt Disney Concert Hall", "Transamerica Pyramid",
        "Gherkin", "Walkie Talkie", "Leadenhall Building", "Petronas Twin Towers", "U.S. Bank Tower",
        "Aon Center", "Citigroup Center", "Trump Tower", "MetLife Building", "Time Warner Center"
    ],
    "landmark": [
        "Eiffel Tower", "Statue of Liberty", "Great Wall of China", "Taj Mahal", "Colosseum",
        "Machu Picchu", "Christ the Redeemer", "Big Ben", "Sydney Opera House", "Pyramids of Giza",
        "Stonehenge", "Angkor Wat", "Mount Rushmore", "Golden Gate Bridge", "Acropolis of Athens",
        "Sagrada Familia", "Neuschwanstein Castle", "Brandenburg Gate", "Petra", "Alhambra",
        "Louvre Museum", "Buckingham Palace", "Notre-Dame Cathedral", "Hagia Sophia", "Leaning Tower of Pisa",
        "Versailles Palace", "Mont Saint-Michel", "Chichen Itza", "Moai Statues of Easter Island", "Burj Khalifa",
        "CN Tower", "Kremlin", "Red Square", "Forbidden City", "Potala Palace", "Palace of Westminster",
        "St. Peter's Basilica", "Uffizi Gallery", "Rijksmuseum", "Hermitage Museum"
    ],
    "stadium": [
        "Madison Square Garden", "Wembley Stadium", "Camp Nou", "Yankee Stadium",
        "Old Trafford", "Allianz Arena", "San Siro", "Maracanã Stadium", "Rose Bowl", "Tokyo Dome",
        "Mercedes-Benz Stadium", "AT&T Stadium", "Anfield", "Stamford Bridge", "Celtic Park",
        "Signal Iduna Park", "Santiago Bernabéu Stadium", "Estadio Azteca", "FNB Stadium", 
        "Gelsenkirchen Stadium", "Emirates Stadium"
    ],
    
    # Products & Technology
    "product": [
        "iPhone", "MacBook", "iPad", "Apple Watch", "AirPods", "Tesla Model S", "PlayStation 5",
        "Xbox Series X", "Nintendo Switch", "Samsung Galaxy S23", "Google Pixel 7", "Dell XPS 13",
        "HP Spectre x360", "Sony WH-1000XM5", "Bose QuietComfort 45", "Kindle Paperwhite",
        "GoPro HERO10", "Fitbit Charge 5", "DJI Mavic Air 2", "Ring Video Doorbell",
        "Nest Thermostat", "Roku Streaming Stick+", "Chromecast with Google TV", "Apple TV 4K", "Amazon Echo Dot",
        "Google Nest Hub", "Samsung QLED TV", "LG OLED TV", "Sonos One", "NVIDIA GeForce RTX 4090",
        "AMD Radeon RX 7900 XTX", "Intel Core i9-13900K", "Corsair Vengeance RAM", "Samsung 980 Pro SSD"
    ],
    "software": [
        "Windows", "macOS", "Microsoft Office", "Adobe Photoshop", "Slack", "Zoom", "Notion",
        "Trello", "Asana", "Visual Studio Code", "IntelliJ IDEA", "PyCharm", "Eclipse", "GitHub Desktop",
        "Docker", "Kubernetes", "Jira", "Confluence", "Figma", "Sketch", "Final Cut Pro", "Logic Pro",
        "Ableton Live", "Pro Tools", "Blender", "AutoCAD", "MATLAB", "SPSS", "Tableau",
        "Power BI", "Salesforce CRM", "SAP ERP", "QuickBooks", "Xero", "WordPress", "Drupal",
        "Joomla", "Magento", "Shopify","Wix", "Squarespace"
    ],
    "app": [
        "Instagram", "TikTok", "WhatsApp", "Snapchat", "Uber", "Spotify", "Netflix",
        "YouTube", "Facebook", "Twitter", "Reddit", "Pinterest", "LinkedIn", "Discord",
        "Telegram", "Signal", "Zoom", "Google Maps", "Waze", "Dropbox", "Evernote",
        "Duolingo", "Headspace", "Calm", "Strava", "Fitbit", "MyFitnessPal",
        "Venmo", "Cash App", "Robinhood", "Coinbase", "Airbnb", "DoorDash",
        "Grubhub", "Postmates", "Yelp", "TripAdvisor", "Hulu", "Disney+"
    ],
    "game": [
        "Minecraft", "Fortnite", "League of Legends", "Call of Duty", "Grand Theft Auto V",
        "The Legend of Zelda", "Super Mario", "Pokemon", "FIFA", "Elden Ring",
        "Overwatch", "Apex Legends", "Valorant", "Among Us", "Roblox", "Animal Crossing", "Cyberpunk 2077",
        "The Witcher 3", "God of War", "Halo Infinite", "Assassin's Creed Valhalla", "Dota 2", "Counter-Strike: Global Offensive", "Rocket League", "Fall Guys", "Genshin Impact",
        "Hades", "Stardew Valley", "Dead by Daylight", "Terraria", "Subnautica", "Dark Souls III"
    ],
    "movie": [
        "Avatar", "Titanic", "Avengers: Endgame", "The Dark Knight", "Inception",
        "Interstellar", "The Matrix", "Star Wars", "Jurassic Park", "The Godfather",
        "Pulp Fiction", "Forrest Gump", "The Shawshank Redemption", "Gladiator", "The Lion King",
        "Frozen", "Toy Story", "Finding Nemo", "The Avengers", "Black Panther",
        "Spider Man", "Iron Man", "Captain America", "Thor", "Hulk"
    ],
    "book": [
        "Harry Potter", "The Lord of the Rings", "A Song of Ice and Fire", "The Da Vinci Code",
        "Sapiens", "Atomic Habits", "The Lean Startup", "Zero to One",
        "Thinking, Fast and Slow", "The Subtle Art of Not Giving a F*ck", "Educated", "Becoming",
        "The Alchemist", "1984", "To Kill a Mockingbird", "The Great Gatsby", "Moby Dick",
        "War and Peace", "Pride and Prejudice", "The Catcher in the Rye", "The Hobbit"
    ],
    "music_album": [
        "Thriller", "The Dark Side of the Moon", "Abbey Road", "Back in Black",
        "1989", "25", "Divide", "Scorpion", "Lemonade", "Future Nostalgia",
        "When We All Fall Asleep, Where Do We Go?", "Fine Line", "Justice",
        "Blonde", "DAMN.", "To Pimp a Butterfly", "Channel Orange", "1984", "Purple Rain", "Born in the U.S.A.",
        "Rumours", "Hotel California", "Led Zeppelin IV", "Sgt. Pepper's Lonely Hearts Club Band", "Appetite for Destruction",
        "Nevermind", "OK Computer", "The Wall", "A Night at the Opera", "The Joshua Tree"
    ],
    "programlang": [
        "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust", "Swift", "Kotlin",
        "Ruby", "PHP", "HTML", "CSS", "SQL", "R", "MATLAB", "Perl", "Lua", "Dart", "Scala",
        "Haskell", "Elixir", "Clojure", "Objective-C", "Shell Scripting",
        "PowerShell", "Visual Basic", "Fortran", "COBOL", "Assembly Language",
        "Groovy", "F#", "Erlang", "Julia", "Ada", "Prolog", "Scheme", "Lisp"
    ],
    "framework": [
        "React", "Angular", "Vue.js", "Django", "Flask", "Spring", "TensorFlow", "PyTorch",
        "Ruby on Rails", "Laravel", "ASP.NET", "Express.js", "Next.js", "Nuxt.js",
        "Svelte", "Ember.js", "Bootstrap", "Tailwind CSS", "jQuery", "Redux",
        "Keras", "Hadoop", "Spark", "Cordova", "Ionic", "Xamarin", "Flutter",
        "Electron", "Gatsby", "GraphQL", "Apollo Client", "NestJS", "FastAPI"
    ],
    "database": [
        "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "Oracle Database",
        "Microsoft SQL Server", "SQLite", "Cassandra", "Firebase", "DynamoDB", "MariaDB",
        "CouchDB", "Neo4j", "InfluxDB", "TimescaleDB", "HBase", "RethinkDB", "CockroachDB", "Memcached",
        "Amazon Aurora", "Google BigQuery", "Snowflake", "IBM Db2", "Teradata",
        "Amazon Redshift", "Google Cloud Spanner", "Azure Cosmos DB"
    ],
    "ai_model": [
        "GPT-4", "ChatGPT", "Claude", "Gemini", "LLaMA", "DALL-E", "Midjourney", "Stable Diffusion",
        "BERT", "RoBERTa", "T5", "Whisper", "PaLM", "ERNIE", "XLNet", "ELECTRA", "GPT-3", "Codex",
        "DINO", "CLIP", "VGG16", "ResNet50", "YOLOv5", "U-Net", "CycleGAN", "DeepLab", "Fast R-CNN",
        "Mask R-CNN", "Transformer", "AlexNet", "InceptionV3", "MobileNet", "EfficientNet", "Swin Transformer",
        "DeBERTa", "ALBERT", "DistilBERT", "Turing-NLG", "ERNIE-GEN", "GShard"
    ],
    "os": [
        "Windows 11", "macOS Sonoma", "Linux", "Ubuntu", "Android", "iOS",
        "Fedora", "Debian", "Red Hat Enterprise Linux", "CentOS", "Arch Linux", "Kali Linux",
        "Chrome OS", "FreeBSD", "OpenBSD", "Solaris", "Gentoo Linux", "Manjaro", "Zorin OS", "Pop!_OS"
    ],
    "cryptocurrency": [
        "Bitcoin", "Ethereum", "Solana", "Cardano", "Dogecoin", "XRP",
        "Polkadot", "Litecoin", "Chainlink", "Uniswap", "Avalanche", "Terra", "Algorand",
        "Cosmos", "VeChain", "Filecoin", "Stellar", "Tezos", "Aave", "Compound", "SushiSwap",
        "PancakeSwap", "Theta", "Zcash", "Dash", "Monero", "EOS", "Tron", "Neo", "Maker", "Yearn.finance",
        "Curve DAO", "Balancer", "Ren", "0x", "Basic Attention Token"
    ],
    
    # Events & Awards
    "event": [
        "CES", "WWDC", "Google I/O", "AWS re:Invent", "Mobile World Congress",
        "E3 Expo", "Comic-Con", "SXSW", "IFA Berlin", "Dreamforce", "VivaTech", "TechCrunch Disrupt", "Slush", "Web Summit", "Collision Conference",
        "RSA Conference", "Black Hat Conference", "DEF CON", "GDC", "PAX", "BlizzCon",
        "Gamescom", "Tokyo Game Show", "Paris Fashion Week", "New York Fashion Week",
        "London Fashion Week", "Milan Fashion Week", "Berlin Fashion Week", "Cannes Film Festival", "Sundance Film Festival",
        "Venice Film Festival", "Toronto International Film Festival", "Berlin International Film Festival", "Tribeca Film Festival",
        "SXSW Film Festival", "Annecy International Animated Film Festival", "Telluride Film Festival", "Locarno Film Festival"
    ],
    "conference": [
        "TED", "Davos Forum", "NeurIPS", "ICML", "CVPR", "ACL",
        "EMNLP", "ICLR", "AAAI", "KDD", "SIGGRAPH", "CHI", "WWW Conference", "ISWC", "UAI",
        "ICRA", "IROS", "RSS", "ECCV", "ICCV", "NAACL", "COLING",
        "WSDM", "CIKM", "ICDE", "VLDB", "SIGMOD"
    ],
    "competition": [
        "Olympics", "World Cup", "Super Bowl", "Wimbledon", "Tour de France",
        "FIFA World Cup", "UEFA Champions League", "NBA Finals", "Stanley Cup", "Cricket World Cup",
        "Rugby World Cup", "Ashes Series", "Copa America", "Indian Premier League", "La Liga",
        "Serie A", "Bundesliga", "Ligue 1", "MLS Cup", "FA Cup",
        "CFL Grey Cup", "NHL Winter Classic", "All England Open Badminton Championships", "World Snooker Championship", "PGA Championship"
    ],
    "award": [
        "Nobel Prize", "Academy Award", "Grammy Award", "Emmy Award", "Pulitzer Prize",
        "Turing Award", "Fields Medal", "Golden Globe", "BAFTA",
        "Cannes Palme d'Or", "Tony Award", "Booker Prize", "National Medal of Technology and Innovation", "Lasker Award",
        "Pritzker Architecture Prize", "Right Livelihood Award", "Wolf Prize", "Sakharov Prize", "Ig Nobel Prize",
        "Hugo Award", "Nebula Award", "Saturn Award", "Critics' Choice Award", "Directors Guild of America Award"
    ],
    "tv_show": [
        "Game of Thrones", "Breaking Bad", "Stranger Things", "The Office", "Friends",
        "The Mandalorian", "The Crown", "Westworld", "The Witcher", "Black Mirror",
        "The Simpsons", "Rick and Morty", "The Big Bang Theory", "Sherlock", "House of Cards",
        "Narcos", "Ozark", "Better Call Saul", "Fargo", "True Detective", "Mindhunter",
        "Chernobyl", "The Handmaid's Tale", "Peaky Blinders", "Succession", "The Boys"
    ],
    
    # Time
    "date": ["January 1, 2024", "December 25, 2023", "July 4, 2022", "November 11, 2021", "October 31, 2020","March 15, 2023",],
    "year": ["2024", "2023", "2022", "2021", "2020", "2019", "2018", "2017", "2016", "2015"],
    "month": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    "century": ["21st century", "20th century", "19th century", "18th century", "17th century", "16th century", "15th century", "14th century", "13th century", "12th century"],


    # === FINANCIAL & NUMERIC ===
    "money": [
        "1 billion dollars", "$44 billion", "100 million euros", "£500,000", 
        "10.5 billion USD", "50 million THB", "net worth of $200B",
        "raised $300M in funding", "valued at €2.3B", "acquired for $1.2 billion",
        "market cap of £1 trillion", "annual revenue of $150 million",
        "profit of ¥5 billion", "investment of ₹750 million",
        "funding round of $25 million", "IPO valued at $3 billion",
        
    ],
    "percent": [
        "15%", "51 percent", "0.5%", "99.9%", "a quarter", "ten percent",
        "half", "three quarters", "eighty percent", "sixty five percent",
        "twelve point five percent", "ninety nine percent", "four percent",
        "seventy two percent", "eleven percent", "forty four percent",
        "sixty percent", "thirty three percent", "eighty five percent",
        "fifty percent",
    ],
    "stock_symbol": [
        "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "NFLX",
        "INTC", "AMD", "ORCL", "CSCO", "IBM", "ADBE", "CRM", "PYPL", "UBER",
        "LYFT", "SQ", "SHOP", "TWTR", "SNAP", "ZM", "DOCU", "ROKU", "SPOT",
        "BIDU", "JD", "BABA", "TCEHY", "PDD", "NTES", "WB", "DIS", "V", "MA", "JPM", "BAC",
        "WFC", "C", "GS", "MS", "AXP", "USB", "TD", "RY", "BNS", "HSBC",
        "DB", "CS", "BARC", "RBS", "BNP", "ING", "SAN", "BBVA", "ITUB"
    ],

    # === DIGITAL & INFRASTRUCTURE ===
    "url": [
        "https://www.openai.com", "www.google.com", "github.com/trending", 
        "apple.co/support", "https://t.co/xyz123",
        "linkedin.com/in/johndoe", "https://news.ycombinator.com", "medium.com/@username", 
        "https://stackoverflow.com/questions", "bit.ly/3xyzabc","www.reddit.com/r/programming"
    ],
    "email": [
        "contact@tesla.com", "support@apple.com", "ceo@microsoft.com", 
        "info@un.org", "admin@stanford.edu",
        "help@openai.com", "info@github.com", "contact@reddit.com", "support@linkedin.com",
        "service@twitter.com", "contact@facebook.com"
    ],
    "phone_number": [
        "+1-555-0199", "02-123-4567", "+44 20 7946 0958", "1-800-APPLE",
        "+81-3-1234-5678", "+91-98765-43210", "03-4567-8901", "+61-2-9876-5432",
        "+49-30-123456", "+33-1-2345-6789", "+86-10-1234-5678", "+7-495-123-4567",
        "+34-91-123-4567", "+39-06-1234-5678", "+55-11-91234-5678",
        "+27-11-123-4567", "+82-2-1234-5678", "+65-6123-4567", "+64-9-123-4567", "+48-22-123-4567",
        "+46-8-123-4567", "+31-20-123-4567", "+41-44-123-4567", "+352-26-123-456",
        "+353-1-123-4567", "+420-2-1234-5678", "+386-1-123-4567", "+421-2-1234-5678",
        "+30-21-1234-5678"
    ],

    # === LEGAL & MEDICAL ===
    "legal_document": [
        "GDPR", "Section 301", "Article 50", "The Constitution", 
        "Patent Act", "Digital Millennium Copyright Act",
        "Freedom of Information Act", "Health Insurance Portability and Accountability Act",
        "Sarbanes-Oxley Act", "Dodd-Frank Act", "Consumer Protection Act",
        "Civil Rights Act", "Clean Air Act", "Affordable Care Act", "Patriot Act",
        "Foreign Corrupt Practices Act", "Family Educational Rights and Privacy Act",
        "Fair Labor Standards Act", "Truth in Lending Act", "Electronic Communications Privacy Act",

    ],
    "disease": [
        "COVID-19", "Diabetes", "Alzheimer's", "Influenza", "Hypertension", "Cancer",
        "Asthma", "Arthritis", "Depression", "HIV/AIDS", "Tuberculosis", "Malaria",
        "Ebola", "Zika virus", "Dengue fever", "Cholera", "Measles", "Mumps", "Rubella",
        "Hepatitis B", "Hepatitis C", "Cystic fibrosis", "Parkinson's disease", "Multiple sclerosis",
        "Lupus", "Crohn's disease", "Ulcerative colitis", "Psoriasis", "Anemia", "Osteoporosis",
        "Glaucoma", "Cataracts", "Migraine", "Epilepsy", "Schizophrenia", "Bipolar disorder",
        "Obsessive-compulsive disorder", "Post-traumatic stress disorder", "Autism spectrum disorder",
        "Attention deficit hyperactivity disorder"
    ],
    "medicine": [
        "Paracetamol", "Insulin", "Pfizer vaccine", "Aspirin", "Ibuprofen",
        "Amoxicillin", "Metformin", "Atorvastatin", "Omeprazole", "Lisinopril",
        "Levothyroxine", "Albuterol", "Simvastatin", "Losartan", "Gabapentin",
        "Hydrochlorothiazide", "Sertraline", "Furosemide", "Zolpidem", "Prednisone",
        "Citalopram", "Montelukast", "Tramadol", "Clopidogrel", "Tamsulosin", "Fluoxetine",
        "Warfarin", "Rosuvastatin", "Duloxetine", "Ranitidine", "Pantoprazole",
        "Cyclobenzaprine", "Meloxicam", "Allopurinol", "Bupropion", "Carvedilol"
    ],
    "animal": [
        "Lion", "Tiger", "Elephant", "Dog", "Cat", "Eagle", "Shark", "Whale", "Penguin", "Panda",
        "Bear", "Wolf", "Dolphin", "Cheetah", "Giraffe", "Zebra", "Kangaroo", "Koala", "Gorilla"
    ],
    "plant": [
        "Rose", "Oak Tree", "Cactus", "Sunflower", "Bamboo", "Pine", "Tulip", "Orchid", "Maple", "Fern",
        "Lotus", "Cherry Blossom", "Aloe Vera", "Lavender", "Jasmine"
    ],
    "instrument": [
        "Guitar", "Piano", "Violin", "Drums", "Flute", "Saxophone", "Trumpet", "Cello", "Harp", "Clarinet",
        "Trombone", "Harmonica", "Ukulele", "Accordion", "Keyboard"
    ],
    
    # ==========================================================================
    # 🔥 CROSS-RE SPECIFIC ENTITIES (For Zero-Shot Generalization)
    # Labels not in v7 dataset but present in Cross-RE
    # ==========================================================================
    "politicalparty": [
        "Democratic Party", "Republican Party", "Labour Party", "Conservative Party",
        "Liberal Democrats", "Green Party", "Communist Party", "Socialist Party",
        "Libertarian Party", "Independence Party", "Nationalist Party", "Progressive Party",
        "People's Party", "Reform Party", "Freedom Party", "Workers' Party",
        "Social Democratic Party", "Christian Democratic Party", "Pirate Party",
        "En Marche!", "Five Star Movement", "Alternative for Germany", "Podemos",
        "Syriza", "Fidesz", "Law and Justice", "Bharatiya Janata Party", "Indian National Congress"
    ],
    "election": [
        "2024 US Presidential Election", "2020 General Election", "Brexit Referendum",
        "2022 Midterm Elections", "French Presidential Election 2022", "German Federal Election 2021",
        "California Gubernatorial Election", "UK General Election 2019", "Indian General Election 2024",
        "European Parliament Elections", "Brazilian Presidential Election 2022",
        "Australian Federal Election 2022", "Japanese House of Councillors Election",
        "Canadian Federal Election 2021", "South Korean Presidential Election 2022"
    ],
    "band": [
        "The Beatles", "Led Zeppelin", "Pink Floyd", "The Rolling Stones", "Queen",
        "Nirvana", "Metallica", "U2", "Coldplay", "Radiohead", "Foo Fighters",
        "Red Hot Chili Peppers", "Guns N' Roses", "AC/DC", "Green Day", "Oasis",
        "The Who", "Fleetwood Mac", "The Beach Boys", "R.E.M.", "Bon Jovi",
        "Depeche Mode", "New Order", "The Cure", "Iron Maiden", "Black Sabbath",
        "BTS", "BLACKPINK", "EXO", "TWICE", "Stray Kids", "NCT"
    ],
    "musicalartist": [
        "Michael Jackson", "Prince", "Madonna", "David Bowie", "Elton John",
        "Whitney Houston", "Mariah Carey", "Celine Dion", "Adele", "Ed Sheeran",
        "The Weeknd", "Taylor Swift", "Beyoncé", "Drake", "Kendrick Lamar",
        "Frank Ocean", "Billie Eilish", "Post Malone", "Dua Lipa", "Harry Styles",
        "Bad Bunny", "J Balvin", "Shakira", "Daddy Yankee", "Rosalía",
        "RM", "V", "Jungkook", "Lisa", "Jennie"
    ],
    "album": [
        "Thriller", "Abbey Road", "The Dark Side of the Moon", "Rumours", "Back in Black",
        "Hotel California", "Led Zeppelin IV", "Appetite for Destruction", "Nevermind", "OK Computer",
        "Kind of Blue", "The Wall", "Born to Run", "Purple Rain", "Blonde on Blonde",
        "1989", "Lemonade", "To Pimp a Butterfly", "My Beautiful Dark Twisted Fantasy", "Reputation",
        "MAP OF THE SOUL: 7", "Love Yourself: Tear", "Born Pink", "Midnights", "Renaissance"
    ],
    "song": [
        "Bohemian Rhapsody", "Imagine", "Hey Jude", "Smells Like Teen Spirit", "Stairway to Heaven",
        "Like a Rolling Stone", "Yesterday", "What's Going On", "Respect", "Good Vibrations",
        "Johnny B. Goode", "Superstition", "Billie Jean", "Purple Haze", "Light My Fire",
        "Shape of You", "Blinding Lights", "Rolling in the Deep", "Uptown Funk", "Despacito",
        "Gangnam Style", "Dynamite", "Butter", "How You Like That", "Kill Bill"
    ],
    "musicgenre": [
        "Rock", "Pop", "Hip Hop", "R&B", "Jazz", "Classical", "Electronic", "Country",
        "Blues", "Reggae", "Metal", "Punk", "Soul", "Funk", "Folk", "Disco",
        "Techno", "House", "Dubstep", "Trap", "K-Pop", "J-Pop", "Latin Pop",
        "Indie Rock", "Alternative", "Grunge", "Post-Rock", "Progressive Rock"
    ],
    "musicalinstrument": [
        "Electric Guitar", "Acoustic Guitar", "Bass Guitar", "Grand Piano", "Synthesizer",
        "Drum Kit", "Violin", "Cello", "Saxophone", "Trumpet", "Flute",
        "Harmonica", "Banjo", "Mandolin", "Trombone", "French Horn",
        "Clarinet", "Oboe", "Harp", "Marimba", "Vibraphone"
    ],
    "writer": [
        "William Shakespeare", "Jane Austen", "Charles Dickens", "Ernest Hemingway",
        "F. Scott Fitzgerald", "Mark Twain", "Virginia Woolf", "James Joyce",
        "George Orwell", "Franz Kafka", "Leo Tolstoy", "Fyodor Dostoevsky",
        "Gabriel García Márquez", "Toni Morrison", "Salman Rushdie", "Kazuo Ishiguro",
        "Haruki Murakami", "J.K. Rowling", "Stephen King", "George R.R. Martin"
    ],
    "poem": [
        "The Waste Land", "The Raven", "Howl", "Paradise Lost", "The Divine Comedy",
        "Leaves of Grass", "The Canterbury Tales", "Beowulf", "Odyssey", "Iliad",
        "Sonnet 18", "Ode to a Nightingale", "The Love Song of J. Alfred Prufrock",
        "Do Not Go Gentle into That Good Night", "Still I Rise", "The Road Not Taken"
    ],
    "literarygenre": [
        "Science Fiction", "Fantasy", "Mystery", "Thriller", "Romance",
        "Horror", "Historical Fiction", "Literary Fiction", "Young Adult",
        "Crime Fiction", "Dystopian", "Magical Realism", "Gothic Fiction",
        "Satire", "Tragedy", "Epic Poetry", "Memoir", "Biography"
    ],
    "academicjournal": [
        "Nature", "Science", "Cell", "The Lancet", "New England Journal of Medicine",
        "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        "Journal of Machine Learning Research", "ACM Computing Surveys",
        "Physical Review Letters", "Chemical Reviews", "Proceedings of the National Academy of Sciences",
        "Journal of the American Chemical Society", "Angewandte Chemie", "Advanced Materials"
    ],
    "researcher": [
        "Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio", "Andrew Ng", "Fei-Fei Li",
        "Demis Hassabis", "Ilya Sutskever", "Ian Goodfellow", "Andrej Karpathy",
        "Daphne Koller", "Michael I. Jordan", "Christopher Manning", "Jitendra Malik",
        "Kaiming He", "Ross Girshick", "Alex Krizhevsky", "Karen Simonyan"
    ],
    "discipline": [
        "Machine Learning", "Artificial Intelligence", "Computer Vision", "Natural Language Processing",
        "Robotics", "Quantum Computing", "Bioinformatics", "Computational Biology",
        "Theoretical Physics", "Organic Chemistry", "Molecular Biology", "Neuroscience",
        "Astrophysics", "Genetics", "Econometrics", "Political Science"
    ],
    "field": [
        "Deep Learning", "Reinforcement Learning", "Transfer Learning", "Few-Shot Learning",
        "Computer Graphics", "Human-Computer Interaction", "Distributed Systems",
        "Cryptography", "Information Retrieval", "Recommender Systems",
        "Speech Recognition", "Image Segmentation", "Object Detection"
    ],
    "theory": [
        "Theory of Relativity", "Quantum Mechanics", "Evolution by Natural Selection",
        "Information Theory", "Game Theory", "Chaos Theory", "String Theory",
        "Big Bang Theory", "Germ Theory of Disease", "Plate Tectonics"
    ],
    "task": [
        "Image Classification", "Object Detection", "Semantic Segmentation",
        "Named Entity Recognition", "Relation Extraction", "Question Answering",
        "Machine Translation", "Text Summarization", "Sentiment Analysis",
        "Speech Recognition", "Text Generation", "Knowledge Graph Construction"
    ],
    "metrics": [
        "F1 Score", "Precision", "Recall", "Accuracy", "AUC-ROC",
        "BLEU Score", "ROUGE Score", "Perplexity", "Mean Average Precision",
        "Intersection over Union", "Word Error Rate", "Cross-Entropy Loss"
    ],
    "algorithm": [
        "Gradient Descent", "Backpropagation", "Adam Optimizer", "Dropout",
        "Batch Normalization", "Attention Mechanism", "Transformer Architecture",
        "Convolutional Neural Network", "Recurrent Neural Network", "LSTM",
        "ResNet", "BERT", "GPT", "YOLO", "U-Net", "GAN"
    ],
    "chemicalcompound": [
        "Water", "Carbon Dioxide", "Methane", "Ethanol", "Glucose",
        "Sodium Chloride", "Sulfuric Acid", "Ammonia", "Acetylsalicylic Acid",
        "Penicillin", "Insulin", "Caffeine", "Nicotine", "Morphine"
    ],
    "chemicalelement": [
        "Hydrogen", "Oxygen", "Carbon", "Nitrogen", "Iron",
        "Gold", "Silver", "Copper", "Platinum", "Uranium",
        "Helium", "Neon", "Argon", "Lithium", "Sodium"
    ],
    "protein": [
        "Hemoglobin", "Insulin", "Collagen", "Keratin", "Myosin",
        "Actin", "Albumin", "Antibody", "Enzyme", "Receptor Protein",
        "p53", "BRCA1", "Spike Protein", "Cas9", "GFP"
    ],
    "enzyme": [
        "DNA Polymerase", "RNA Polymerase", "Lipase", "Amylase", "Protease",
        "ATP Synthase", "Helicase", "Ligase", "Kinase", "Phosphatase",
        "CRISPR-Cas9", "Restriction Enzyme", "Reverse Transcriptase"
    ],
    "astronomicalobject": [
        "Sun", "Moon", "Mars", "Jupiter", "Saturn", "Venus", "Mercury",
        "Milky Way", "Andromeda Galaxy", "Proxima Centauri", "Alpha Centauri",
        "Black Hole M87", "Sagittarius A*", "Crab Nebula", "Orion Nebula",
        "Halley's Comet", "Asteroid Bennu", "Pluto", "Europa", "Titan"
    ],
    "misc": [
        "Project Alpha", "Initiative X", "Phase Two", "Category A", "Type B",
        "unknown entity", "unspecified object", "classified item", "redacted name",
        "the entity", "the subject", "the target", "reference point"
    ],
}

# === CHINESE ENTITIES ===
ZH_ENTITIES = {
    "person": [
        "張忠謀", "郭台銘", "馬雲", "馬化騰", "李彥宏", "任正非", "雷軍", "劉強東", "張一鳴", "黃錚",
        "陳志明", "林雅婷", "張偉", "王芳", "李軍", 
        "劉洋", "楊秀英", "黃志偉", "吳建國", "趙麗", 
        "周傑", "徐明", "孫娜", "馬超", "朱曉明", 
        "胡平", "郭建華", "何敏", "高山", "羅偉",
        "陳怡君", "林志豪", "張淑芬", "王建宏", "李美玲", 
        "劉冠宇", "楊宗翰", "黃心怡", "趙家豪", "吳雅雯", 
        "許家瑋", "鄭婷婷", "謝欣怡", "曾國強", "賴建邦", 
        "蔡佩珊", "梁文傑", "宋小梅", "鄧大為", "馮志強", 
        "彭偉文", "潘志明", "袁小妹", "于凱文", "蔣偉國", 
        "沈大偉", "余志平", "盧俊義", "葉大山", "魏小寶"
    ],
    "politician": [
        "蔡英文", "習近平", "李克強", "王毅", "賴清德", "韓國瑜", "朱立倫", "陳水扁", "馬英九", "林鄭月娥",
        "汪洋", "栗戰書", "王滬寧", "趙樂際", "韓正", "劉鶴", "張高麗", "王岐山", "李源潮", "張德江", "俞正聲",
        "胡錦濤", "溫家寶", "江澤民", "朱鎔基", "李鵬", "鄧小平", "毛澤東", "周恩來", "蔣介石"
    ],
    "scientist": [
        "屠呦呦", "楊振寧", "李政道", "丁肇中", "錢學森", "袁隆平", "高錕", "南仁東", "潘建偉", "施一公",
        "吳恩達", "李飛飛", "張亞勤", "王小謨", "陳建民", "周忠和", "張首晟", "俞敏洪", "曹雪濤", "陳薇",
        "施一公", "顧秉林", "葉培建", "朱棣文", "吳健雄", "李四光", "華羅庚", "錢三強", "竺可楨", "丁肇中"
    ],
    "athlete": [
        "姚明", "劉翔", "李娜", "蘇炳添", "谷愛凌", "朱婷", "林書豪", "王濛", "張繼科", "孫楊",
        "陳夢", "馬龍", "張怡寧", "林丹", "郭晶晶", "張虹", "吳敏霞", "劉詩雯", "陳若琳", "馮坤",
        "張琳芃", "武磊", "王霜", "孫雯", "李小鵬", "劉璇", "楊威", "張娜", "陳一冰", "周洋",
    ],
    "musician": [
        "周杰倫", "林俊傑", "蔡依林", "張惠妹", "五月天", "鄧紫棋", "王力宏", "張學友", "陳奕迅", "蕭敬騰",
        "李榮浩", "楊丞琳", "田馥甄", "張韶涵", "林宥嘉", "吳青峰", "陳綺貞", "張震嶽", "羅大佑", "費玉清",
        "鄧麗君", "張國榮", "梅艷芳", "譚詠麟", "Beyond", "張信哲", "王菲", "陳百強", "許冠傑", "林子祥"
    ],
    "actor": [
        "成龍", "李連杰", "周潤發", "梁朝偉", "劉德華", "章子怡", "鞏俐", "范冰冰", "黃渤", "吳京", "楊冪", "迪麗熱巴", "劉亦菲", "周星馳", "張曼玉",
        "林青霞", "張國榮", "周迅", "孫儷", "楊紫瓊", "劉嘉玲", "張涵予", "陳道明", "黃曉明", "胡歌", "井柏然", "趙薇", "范偉", "徐峥", "吳彥祖",
        "陳坤", "吳秀波", "張譯", "李冰冰", "劉青雲", "古天樂", "吳鎮宇", "張家輝", "任達華", "謝霆鋒"
    ],
    "director": [
        "張藝謀", "李安", "王家衛", "陳凱歌", "馮小剛", "吳宇森", "徐克", "姜文", "陳可辛", "賈樟柯",
        "周星馳", "吳京", "林超賢", "韓寒", "寧浩", "張一白", "陳思誠", "葉偉信", "杜琪峯", "彭浩翔", "許鞍華",
        "陳嘉上", "劉偉強", "王晶", "徐崢", "薛曉路", "陳正道", "田壯壯", "吳念真", "李少紅", "陳衍儒"
    ],
    "author": [
        "莫言", "余華", "劉慈欣", "金庸", "瓊瑤", "張愛玲", "巴金", "老舍", "魯迅", "曹雪芹",
        "施耐庵", "吳承恩", "羅貫中", "金庸", "古龍", "梁羽生", "王小波", "韓寒", "郭敬明", "張悅然", "劉震雲",
        "賈平凹", "畢飛宇", "陳忠實", "蘇童", "張炜", "余秋雨", "龍應台", "李敖", "林清玄", "三毛"
    ],
    "company": [
        "台積電", "鴻海", "聯發科", "華碩", "宏碁", "阿里巴巴", "騰訊", "百度", "華為", "小米",
        "京東", "美團", "字節跳動", "拼多多", "滴滴", "網易", "聯想", "比亞迪", "寧德時代",
        "中興通訊", "海康威視", "中國移動", "中國電信", "中國聯通", "中國石油", "中國石化", "中國建築", "中國鐵建", "中國中鐵",
        "中國平安", "中國人壽", "中國銀行", "工商銀行", "建設銀行", "農業銀行", "交通銀行", "招商銀行", "光大銀行", "浦發銀行",
        "中國太平洋保險", "中國人保", "中國郵政集團", "中國煙草總公司", "中國航天科技集團", "中國航天科工集團"
    ],
    "startup": [
        "商湯科技", "曠視科技", "依圖科技", "雲從科技", "字節跳動", "快手", "拼多多", "滴滴出行", "美團點評", "小紅書",
        "B站", "知乎", "愛奇藝", "蔚來汽車", "小鵬汽車", "理想汽車", "貝殼找房", "車好多", "趣店", "歡聚時代", "映客直播",
        "陌陌", "一直播", "火山小視頻", "微播易", "有贊", "微盟", "拉勾網", "獵聘網", "Boss直聘", "脈脈"
    ],
    "university": [
        "北京大學", "清華大學", "復旦大學", "上海交通大學", "浙江大學",
        "台灣大學", "成功大學", "交通大學", "中央大學", "中山大學",
        "南京大學", "武漢大學", "華中科技大學", "西安交通大學", "同濟大學", "南開大學", "天津大學", "吉林大學", "東北大學", "哈爾濱工業大學",
        "中南大學", "湖南大學", "廈門大學", "四川大學", "重慶大學", "華南理工大學", "中科大", "蘭州大學", "西北工業大學", "北京航空航天大學"
    ],
    "sports_team": [
        "中國國家足球隊", "中華台北隊", "廣州恆大", "北京國安", "上海上港",
        "深圳隊", "武漢卓爾", "天津泰達", "山東魯能", "江蘇蘇寧",
        "台灣職棒兄弟象", "統一獅", "富邦悍將", "中信兄弟", "味全龍",
        "台北富邦勇士", "新北國王", "高雄鋼鐵人", "桃園領航猿", "台中太陽",
        "北京首鋼隊", "廣東宏遠隊", "遼寧飛豹隊", "新疆廣匯隊", "山東西王隊"
    ],
    "bank": [
        "中國工商銀行", "中國建設銀行", "中國銀行", "國泰世華銀行", "中國信託",
        "台新銀行", "土地銀行", "合作金庫銀行", "華南銀行", "第一商業銀行",
        "花旗銀行", "渣打銀行", "匯豐銀行", "星展銀行", "永豐銀行",
        "台灣土地銀行", "兆豐國際商業銀行", "台灣中小企業銀行", "台灣農業金庫", "台灣銀行"
    ],
    "city": [
        "台北", "新竹", "台中", "高雄", "北京", "上海", "深圳", "杭州", "廣州", "成都",
        "香港", "澳門", "南京", "武漢", "西安", "青島", "廈門", "大連", "蘇州", "天津", "重慶",
        "福州", "長沙", "昆明", "南寧", "濟南", "合肥", "鄭州", "石家莊", "太原", "呼和浩特",
        "烏魯木齊", "拉薩", "銀川", "海口", "三亞", "珠海", "廊坊", "唐山", "煙台", "威海",
        "紹興", "嘉興", "湖州", "金華", "衢州", "舟山", "臺州", "溫州"
    ],
    "country": [
        "中國", "台灣", "日本", "韓國", "新加坡", "美國", "加拿大", "澳大利亞", "英國", "法國",
        "德國", "俄羅斯", "印度", "巴西", "南非", "墨西哥", "印尼", "泰國", "越南", "菲律賓",
        "馬來西亞", "紐西蘭", "荷蘭", "比利時", "瑞士", "瑞典", "挪威", "丹麥", "芬蘭", "奧地利", "愛爾蘭",
        "葡萄牙", "希臘", "土耳其", "埃及", "阿聯酋", "沙特阿拉伯", "以色列", "卡塔爾", "科威特", "巴林"
    ],
    "product": [
        "微信", "QQ", "抖音", "TikTok", "支付寶", "淘寶", "天貓", "京東商城",
        "華為Mate", "小米手機", "OPPO", "vivo", "比亞迪電動車", "蔚來汽車", "理想汽車", "小鵬汽車", "聯想筆記本", "戴爾電腦", "蘋果iPhone", "三星手機",
        "華碩筆記本", "宏碁電腦", "小熊電器", "美的空調", "海爾冰箱", "格力空調", "海信電視", "TCL手機", "一加手機", "魅族手機",
        "360安全衛士", "搜狗輸入法", "百度地圖", "高德地圖", "滴滴出行", "美團外賣", "餓了麼", "攜程旅行"
    ],
    "movie": [
        "戰狼2", "流浪地球", "哪吒之魔童降世", "長津湖", "紅海行動", "唐人街探案", "我不是藥神", "中國機長", "八佰", "速度與激情：特別行動",
        "復仇者聯盟4：終局之戰", "蜘蛛人：英雄無歸", "侏羅紀世界3", "速度與激情9", "黑寡婦", "星際異攻隊3", "神奇女俠1984", "正義聯盟", "水行俠", "雷神4：愛與雷霆",
        "變形金剛：終極戰士", "死侍2", "X戰警：黑鳳凰", "金剛戰士", "驚奇隊長", "蟻人與黃蜂女", "黑豹", "奇異博士", "銀河護衛隊2", "美國隊長3：英雄內戰"
    ],
    "book": [
        "三體", "活著", "圍城", "紅樓夢", "射雕英雄傳", "天龍八部", "笑傲江湖", "倚天屠龍記", "平凡的世界", "白鹿原",
        "哈利波特系列", "指環王", "冰與火之歌", "達·芬奇密碼", "安娜·卡列尼娜", "戰爭與和平", "悲慘世界", "小王子", "1984",
        "動物農場", "了不起的蓋茨比", "追風箏的人", "解憂雜貨店", "嫌疑人X的獻身", "白夜行", "東野圭吾作品集", "福爾摩斯探案集", "時間簡史", "自私的基因"
    ],
    "award": [
        "金馬獎", "金鐘獎", "金曲獎", "中國電影金雞獎", "華語電影傳媒大獎",
        "百花獎", "香港電影金像獎", "亞洲電影大獎", "中國電視金鷹獎", "中國文學獎",
        "魯迅文學獎", "茅盾文學獎", "曹禺戲劇獎", "華語電影最佳導演獎", "最佳男主角獎", "最佳女主角獎",
        "最佳編劇獎", "最佳攝影獎", "最佳剪輯獎", "最佳音樂獎"
    ],
    "date": ["12月25日", "1月1日", "7月4日", "10月1日113年", "2月14日", "6月1日", "11月11日", "3月8日", "5月1日", "9月10日","112/08/20",
             "111/12/31", "110/10/10", "109/05/20", "108/03/15", "107/07/04"
             ],
    "year": ["2024年", "2023年", "2022年", "2021年", "2020年", "2019年", "2018年", "2017年", "2016年", "2015年"],
    "month": ["一月", "二月", "三月", "四月", "五月", "六月", "七月", "八月", "九月", "十月", "十一月", "十二月",
              "1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"],
    "century": ["21世紀", "20世紀", "19世紀", "18世紀", "17世紀", "16世紀", "15世紀", "14世紀", "13世紀", "12世紀"],
    "location": [
        "故宮", "長城", "西湖", "黃山", "張家界", "九寨溝",
        "桂林", "兵馬俑", "天壇", "頤和園", "峨眉山", "廬山", "泰山", "雲南石林", "鳳凰古城",
        "陽朔", "西雙版納", "青海湖", "拉薩布達拉宮", "新疆天山", "內蒙古草原",
        "黃果樹瀑布", "千島湖", "雲南大理", "雲南麗江", "福建土樓", "武夷山", "廈門鼓浪嶼",
        "泉州古城", "潮州古城", "廣州塔", "深圳灣大橋", "珠海長隆", "澳門大三巴牌坊",
        "香港維多利亞港", "澳門威尼斯人", "香港迪士尼樂園", "上海外灘", "南京路步行街", "東方明珠塔",
        ],
    "animal": ["狮子", "老虎", "大象", "狗", "猫", "鹰", "鲨鱼", "熊猫", "企鹅", "狼"],
    "plant": ["玫瑰", "橡树", "仙人掌", "向日葵", "竹子", "松树", "郁金香", "兰花"],
    "instrument": ["吉他", "钢琴", "小提琴", "鼓", "长笛", "萨克斯", "古筝", "二胡"],
}

# === JAPANESE ENTITIES ===
JA_ENTITIES = {
    "person": [
        "盛田昭夫", "井深大", "本田宗一郎", "豊田喜一郎", "松下幸之助",
        "孫正義", "三木谷浩史", "柳井正", "稲盛和夫", "安藤百福",
        "佐藤 健", "鈴木 一郎", "高橋 誠", "田中 実", "渡辺 裕子",
        "伊藤 恵", "山本 太郎", "中村 さくら", "小林 剛", "加藤 美咲",
        "吉田 拓也", "山田 花子", "佐々木 翔", "山口 智子", "松本 潤一",
        "井上 陽子", "木村 拓哉", "林 健太", "清水 翔太", "山崎 賢人"
    ],
    "scientist": [
        "湯川秀樹", "本庶佑", "山中伸弥", "中村修二", "吉野彰"
    ],
    "athlete": [
        "大谷翔平", "イチロー", "錦織圭", "大坂なおみ", "羽生結弦"
    ],
    "musician": [
        "米津玄師", "宇多田ヒカル", "YOASOBI", "Ado", "藤井風"
    ],
    "actor": [
        "渡辺謙", "真田広之", "役所広司", "新垣結衣", "長澤まさみ"
    ],
    "director": [
        "宮崎駿", "黒澤明", "北野武", "是枝裕和", "新海誠"
    ],
    "author": [
        "村上春樹", "東野圭吾", "川端康成", "三島由紀夫", "芥川龍之介"
    ],
    "company": [
        "ソニー", "トヨタ", "ホンダ", "任天堂", "パナソニック", "日立",
        "東芝", "キヤノン", "富士通", "NEC", "ソフトバンク", "楽天", "ファーストリテイリング"
    ],
    "startup": [
        "メルカリ", "SmartNews", "Preferred Networks", "freee", "UUUM"
    ],
    "university": [
        "東京大学", "京都大学", "大阪大学", "東北大学", "早稲田大学", "慶應義塾大学"
    ],
    "sports_team": [
        "読売ジャイアンツ", "阪神タイガース", "鹿島アントラーズ", "浦和レッズ"
    ],
    "bank": [
        "三菱UFJ銀行", "三井住友銀行", "みずほ銀行", "りそな銀行"
    ],
    "city": [
        "東京", "大阪", "京都", "名古屋", "福岡", "横浜", "神戸", "札幌",
        "広島", "仙台", "川崎", "浜松"
    ],
    "country": [
        "日本", "アメリカ", "中国", "韓国", "フランス", "ドイツ"
    ],
    "product": [
        "PlayStation", "Nintendo Switch", "ウォークマン", "プリウス",
        "ゼルダの伝説", "マリオ", "ポケモン", "ファイナルファンタジー"
    ],
    "movie": [
        "千と千尋の神隠し", "君の名は。", "もののけ姫", "ドライブ・マイ・カー"
    ],
    "book": [
        "ノルウェイの森", "1Q84", "源氏物語", "羅生門"
    ],
    "award": [
        "日本アカデミー賞", "芥川賞", "直木賞", "レコード大賞"
    ],
    "date": ["2024年", "2023年", "2022年", "2021年", "2020年", "2010年", "2000年", "1990年"],
    "location": ["秋葉原", "新宿", "銀座", "道頓堀", "六本木ヒルズ"],
    "animal": ["ライオン", "虎", "象", "犬", "猫", "鷲", "サメ", "パンダ", "ペンギン", "狼"],
    "plant": ["バラ", "オーク", "サボテン", "ひまわり", "竹", "松", "チューリップ", "蘭"],
    "instrument": ["ギター", "ピアノ", "バイオリン", "ドラム", "フルート", "サックス", "三味線", "琴"],
}

# === KOREAN ENTITIES ===
KO_ENTITIES = {
    "person": [
        "이병철", "이건희", "이재용", "정주영", "정몽구", "정의선",
        "김범수", "이해진", "방시혁"
    ],
    "politician": [
        "윤석열", "문재인", "박근혜", "이명박", "김대중"
    ],
    "scientist": [
        "이휘소", "김대중", "황우석"
    ],
    "athlete": [
        "손흥민", "김연아", "박지성", "류현진", "박세리"
    ],
    "musician": [
        "BTS", "BLACKPINK", "아이유", "PSY", "EXO", "TWICE", "NewJeans"
    ],
    "actor": [
        "송강호", "이병헌", "전지현", "손예진", "박서준"
    ],
    "director": [
        "봉준호", "박찬욱", "김기덕", "이창동", "나홍진"
    ],
    "company": [
        "삼성전자", "현대자동차", "SK하이닉스", "LG전자", "포스코",
        "카카오", "네이버", "쿠팡", "하이브", "현대중공업"
    ],
    "startup": [
        "토스", "당근마켓", "야놀자", "무신사", "오늘의집"
    ],
    "university": [
        "서울대학교", "연세대학교", "고려대학교", "KAIST", "포항공과대학교"
    ],
    "sports_team": [
        "손흥민", "전북현대", "울산현대", "FC서울", "두산베어스"
    ],
    "bank": [
        "국민은행", "신한은행", "하나은행", "우리은행", "기업은행"
    ],
    "city": [
        "서울", "부산", "인천", "대구", "광주", "대전", "울산", "수원",
        "성남", "판교", "제주"
    ],
    "country": [
        "대한민국", "미국", "일본", "중국", "영국"
    ],
    "product": [
        "갤럭시", "카카오톡", "네이버", "쿠팡", "배달의민족", "토스",
        "현대차", "기아차", "삼성페이"
    ],
    "movie": [
        "기생충", "올드보이", "괴물", "부산행", "헤어질 결심"
    ],
    "award": [
        "대종상", "청룡영화상", "백상예술대상", "멜론뮤직어워드"
    ],
    "date": ["2024년", "2023년", "2022년", "2021년", "2020년", "2010년", "2000년"],
    "animal": ["사자", "호랑이", "코끼리", "개", "고양이", "독수리", "상어", "판다", "펭귄", "늑대"],
    "plant": ["장미", "참나무", "선인장", "해바라기", "대나무", "소나무", "튤립", "난초"],
    "instrument": ["기타", "피아노", "바이올린", "드럼", "플루트", "색소폰", "가야금", "해금"],
}

# === THAI ENTITIES ===
TH_ENTITIES = {
    "person": [
        "ธนินท์ เจียรวนนท์", "เจริญ สิริวัฒนภักดี", "ชิน โสภณพนิช",
        "วิชัย ศรีวัฒนประภา", "ธนาธร จึงรุ่งเรืองกิจ",
        "สมชาย ใจดี", "สมศรี รักสงบ", "วิชัย มีโชค", "นารี รัตนกุล", "ประเสริฐ สุขใจ",
        "กานดา มั่นคง", "อาทิตย์ แสงสว่าง", "วันเพ็ญ จันทร์ส่อง", "สุชาติ พอเพียง", "มานี มีนา",
        "ปิติ ยินดี", "ชูใจ ใฝ่ดี", "วีระ กล้าหาญ", "สุดา น่ารัก", "สมศักดิ์ ภักดี",
        "รัตนา วงศ์สวัสดิ์", "วิภา งามตา", "ณัฐวุฒิ ภูมิใจ", "กมลวรรณ สดใส", "ธนพล ร่ำรวย"
    ],
    "politician": [
        "ประยุทธ์ จันทร์โอชา", "ทักษิณ ชินวัตร", "ยิ่งลักษณ์ ชินวัตร", "เศรษฐา ทวีสิน"
    ],
    "athlete": [
        "ทัพพ์ แสงสว่าง", "รัชนก อินทนนท์", "สรวีย์ เจริญประเสริฐ"
    ],
    "musician": [
        "ลิซ่า", "แบมแบม", "ใบเฟิร์น", "มาริโอ้", "ณเดชน์"
    ],
    "actor": [
        "โทนี่ จา", "มาริโอ้ เมาเร่อ", "ณเดชน์ คูกิมิยะ", "ใบเฟิร์น พิมพ์ชนก"
    ],
    "company": [
        "เครือเจริญโภคภัณฑ์", "ปตท.", "ธนาคารกรุงเทพ", "เซ็นทรัล",
        "ไทยเบฟเวอเรจ", "ทรู", "AIS", "SCB", "กสิกรไทย"
    ],
    "startup": [
        "Grab Thailand", "Lazada Thailand", "Shopee Thailand", "LINE MAN"
    ],
    "university": [
        "จุฬาลงกรณ์มหาวิทยาลัย", "มหาวิทยาลัยธรรมศาสตร์", "มหาวิทยาลัยมหิดล",
        "มหาวิทยาลัยเกษตรศาสตร์"
    ],
    "bank": [
        "ธนาคารกรุงเทพ", "ธนาคารไทยพาณิชย์", "ธนาคารกสิกรไทย", "ธนาคารกรุงไทย"
    ],
    "city": [
        "กรุงเทพฯ", "เชียงใหม่", "ภูเก็ต", "พัทยา", "หาดใหญ่",
        "ขอนแก่น", "นครราชสีมา", "อุดรธานี"
    ],
    "country": [
        "ประเทศไทย", "สหรัฐอเมริกา", "ญี่ปุ่น", "จีน", "สิงคโปร์"
    ],
    "product": [
        "ทรูมูฟ", "AIS", "DTAC", "LINE", "แกร็บ", "ลาซาด้า", "ช้อปปี้"
    ],
    "movie": [
        "องค์บาก", "ต้มยำกุ้ง", "พี่มาก..พระโขนง", "ฉลาดเกมส์โกง"
    ],
    "award": [
        "สุพรรณหงส์", "นาฏราช", "ตุ๊กตาทอง"
    ],
    "date": ["2024", "2023", "2022", "2021", "2020", "2010", "2000"],
    "animal": ["สิงโต", "เสือ", "ช้าง", "สุนัข", "แมว", "นกอินทรี", "ฉลาม", "วาฬ", "เพนกวิน", "แพนด้า"],
    "plant": ["กุหลาบ", "ต้นโอ๊ก", "กระบองเพชร", "ทานตะวัน", "ไผ่", "สน", "ทิวลิป", "กล้วยไม้"],
    "instrument": ["กีตาร์", "เปียโน", "ไวโอลิน", "กลอง", "ขลุ่ย", "แซกโซโฟน", "ระนาด", "ซออู้"],
}
ZH_ENTITIES.update({
    "person": ZH_ENTITIES["person"] + ["任正非", "孟晚舟", "王傳福", "潘石屹", "董明珠"],
    "company": ZH_ENTITIES["company"] + ["比亞迪", "寧德時代", "中芯國際", "美團", "攜程"],
    "product": ZH_ENTITIES["product"] + ["鴻蒙OS", "支付寶", "文心一言", "小紅書"],
    "money": ["100億人民幣", "5000萬美金", "十億元", "3000萬港幣"],
    "stock_symbol": ["0700.HK", "BABA", "9988.HK", "BIDU", "300750.SZ"],
    "legal_document": ["《中華人民共和國民法典》", "《數據安全法》", "粵港澳大灣區規劃"],
    "university": ZH_ENTITIES["university"] + ["香港大學", "香港科技大學", "澳門大學"]
})
JA_ENTITIES.update({
    "person": JA_ENTITIES["person"] + ["豊田章男", "佐藤恒治", "新浪剛史", "十時裕樹"],
    "company": JA_ENTITIES["company"] + ["キーエンス", "三菱商事", "日本郵政", "ファナック", "任天堂"],
    "product": JA_ENTITIES["product"] + ["ウォークマン", "プリウス", "カローラ", "写ルンです"],
    "money": ["10億円", "5000万ドル", "300兆円", "100万ユーロ"],
    "stock_symbol": ["7203.T", "6758.T", "9984.T", "6861.T"],
    "url": ["https://www.sony.jp", "https://www.toyota.co.jp", "yahoo.co.jp"],
    "location": JA_ENTITIES["location"] + ["秋葉原", "新宿", "銀座", "道頓堀", "六本木ヒルズ"]
})
KO_ENTITIES.update({
    "person": KO_ENTITIES["person"] + ["최태원", "구광모", "신동빈", "장현승"],
    "company": KO_ENTITIES["company"] + ["SK이노베이션", "LG에너지솔루션", "네이버제트", "에ส엠엔터테인먼트"],
    "product": KO_ENTITIES["product"] + ["제네시스", "V3", "라인", "싸이월드"],
    "money": ["1000억 원", "5000만 달러", "십억 원", "100만 유로"],
    "stock_symbol": ["005930.KS", "000660.KS", "035420.KS", "035720.KS"],
    "award": KO_ENTITIES["award"] + ["MAMA 어워즈", "골든디스크어워즈"],
    "city": KO_ENTITIES["city"] + ["송도", "세종시", "창원", "청주"]
})
TH_ENTITIES.update({
    "person": TH_ENTITIES["person"] + [
        "ชูวิทย์ กมลวิศิษฎ์", "นวลพรรณ ล่ำซำ", "อัยยวัฒน์ ศรีวัฒนประภา",
        "สมหมาย ขายดี", "สมชาย มีทรัพย์", "วิภา รัตนไพศาล", "กนกวรรณ แก้วดี"
    ],
    "company": TH_ENTITIES["company"] + [
        "บริษัท ปูนซิเมนต์ไทย จำกัด (มหาชน)", "เครือสหพัฒน์", "กัลฟ์ เอ็นเนอร์จี",
        "ศรีสวัสดิ์", "โอสถสภา", "ไมเนอร์ อินเตอร์เนชั่นแนล"
    ],
    "money": ["1,000 ล้านบาท", "5 หมื่นล้านเหรียญ", "สิบล้านยูโร", "500,000 บาท"],
    "percent": ["ร้อยละ 50", "15 เปอร์เซ็นต์", "0.25%", "สิบเปอร์เซ็นต์"],
    "stock_symbol": ["PTT", "CPALL", "AOT", "SCC", "ADVANC", "KBANK", "SCB"],
    "legal_document": ["พรบ. คุ้มครองข้อมูลส่วนบุคคล (PDPA)", "มาตรา 112", "รัฐธรรมนูญฉบับปี 2560"],
    "url": ["https://www.set.or.th", "https://www.bot.or.th", "pantip.com"],
    "university": TH_ENTITIES["university"] + ["มหาวิทยาลัยมหิดล", "ม.เชียงใหม่", "มก."]
})

# ============================================================================
# TEMPLATE GENERATORS
# ============================================================================

def get_entity(entities_dict: Dict, entity_type: str) -> str:
    """Get a random entity of the given type from the dictionary."""
    # Try exact match first
    if entity_type in entities_dict and entities_dict[entity_type]:
        return random.choice(entities_dict[entity_type])
    
    # Try parent category mapping
    type_mapping = {
        # 🔥 ZERO-SHOT: US vs UK spelling
        "organization": "organization",  # US spelling → UK spelling in entities
        "politician": "person",
        "scientist": "person",
        "artist": "person",
        "athlete": "person",
        "musician": "person",
        "actor": "person",
        "director": "person",
        "author": "person",
        "entrepreneur": "person",
        "engineer": "person",
        "doctor": "person",
        "journalist": "person",
        "chef": "person",
        "startup": "company",
        "nonprofit": "organization",
        "government_agency": "organization",
        "school": "university",
        "hospital": "organization",
        "military": "organization",
        "political_party": "organization",
        "research_institute": "university",
        "museum": "organization",
        "restaurant": "company",
        "hotel": "company",
        "state": "city",
        "continent": "country",
        "region": "location",
        "building": "location",
        "landmark": "location",
        "airport": "location",
        "stadium": "location",
        "park": "location",
        "island": "location",
        "mountain": "location",
        "river": "location",
        "neighborhood": "city",
        "year": "date",
        "month": "date",
        "time": "date",
        "duration": "date",
        "era": "date",
        "century": "date",
        "season": "date",
        "software": "product",
        "hardware": "product",
        "vehicle": "product",
        "food": "product",
        "beverage": "product",
        "medicine": "product",
        "electronics": "product",
        "clothing": "product",
        "cosmetics": "product",
        "framework": "programlang",
        "database": "product",
        "protocol": "product",
        "api": "product",
        "algorithm": "product",
        "os": "software",
        "technology": "product",
        "conference": "event",
        "festival": "event",
        "war": "event",
        "election": "event",
        "disaster": "event",
        "ceremony": "event",
        "artwork": "product",
        "patent": "product",
        "invention": "product",
        "research_paper": "book",
        "degree": "award",
        "title": "award",
        "skill": "product",
        # Financial & Numeric
        "money": "money",          # ถ้ามีคลาสแยกอยู่แล้ว
        "percent": "percent",
        "stock_symbol": "stock_symbol",
        "quantity": "quantity",
        
        # Digital & Infrastructure
        "url": "url",
        "email": "email",
        "phone_number": "phone_number",
        "ip_address": "url",        # fallback ไปที่ url ถ้าไม่มี ip โดยเฉพาะ
        
        # Medical & Legal
        "disease": "disease",
        "legal_document": "legal_document",
        "academic_field": "academic_field",
        "organ": "location",        # fallback อวัยวะไปที่สถานที่/ตำแหน่ง
        
        # ย้ายความสัมพันธ์เดิมบางส่วนให้แม่นยำขึ้น
        "medicine": "medicine",     # เปลี่ยนจาก product เป็น medicine โดยตรง
        "vaccine": "medicine",
        "treatment": "medicine",


        
    }
    
    if entity_type in type_mapping:
        parent_type = type_mapping[entity_type]
        if parent_type in entities_dict and entities_dict[parent_type]:
            return random.choice(entities_dict[parent_type])
    
    # 3. Last Resort Fallback (เจาะจงตามกลุ่ม)
    # ถ้าหา 'money' หรือ 'percent' ไม่เจอ ให้คืนค่าตัวเลขสมมติ
    if entity_type in ["money", "percent", "quantity"]:
        return str(random.randint(1, 1000))
    
    # 4. Fallback ไปที่คลาสมาตรฐาน
    for fallback in ["person", "company", "location", "product"]:
        if fallback in entities_dict and entities_dict[fallback]:
            return random.choice(entities_dict[fallback])
    
    return "Unknown"


# ============================================================
# === TEMPLATES WITH RELATIONS ===
# ============================================================

# English Templates with diverse relations
EN_TEMPLATES = [
    # founder_of + founded_in + located_in
    ("{person} founded {company} in {date} in {city}.",
     [("person", "person"), ("company", "company"), ("date", "date"), ("city", "city")],
     [("person", "company", "founder_of"), ("company", "date", "founded_in"), ("company", "city", "located_in")]),
    
    # ceo_of
    ("{person} is the CEO of {company}.",
     [("person", "person"), ("company", "company")],
     [("person", "company", "ceo_of")]),
    
    ("{person} serves as CEO of {company}, which is headquartered in {city}.",
     [("person", "person"), ("company", "company"), ("city", "city")],
     [("person", "company", "ceo_of"), ("company", "city", "headquartered_in")]),
    
    # developed + released_in
    ("{company} developed {product} in {date}.",
     [("company", "company"), ("product", "product"), ("date", "date")],
     [("company", "product", "developed"), ("product", "date", "released_in")]),
    
    # creator_of
    ("{engineer} created {programlang}.",
     [("engineer", "engineer"), ("programlang", "programlang")],
     [("engineer", "programlang", "creator_of")]),
    
    # author_of
    ("{author} wrote {book}.",
     [("author", "author"), ("book", "book")],
     [("author", "book", "author_of")]),
    
    # director_of
    ("{director} directed {movie}.",
     [("director", "director"), ("movie", "movie")],
     [("director", "movie", "director_of")]),
    
    # starred_in
    ("{actor} starred in {movie}.",
     [("actor", "actor"), ("movie", "movie")],
     [("actor", "movie", "starred_in")]),
    
    # studied_at + graduated_from
    ("{person} studied at {university} and graduated in {date}.",
     [("person", "person"), ("university", "university"), ("date", "date")],
     [("person", "university", "studied_at"), ("person", "date", "graduated_in")]),
    
    # works_at
    ("{person} works at {company} in {city}.",
     [("person", "person"), ("company", "company"), ("city", "city")],
     [("person", "company", "works_at"), ("company", "city", "located_in")]),
    
    # plays_for (athlete)
    ("{athlete} plays for {sports_team}.",
     [("athlete", "athlete"), ("sports_team", "sports_team")],
     [("athlete", "sports_team", "plays_for")]),
    
    # won (award)
    ("{person} won the {award}.",
     [("person", "person"), ("award", "award")],
     [("person", "award", "won")]),
    
    # acquired_by
    ("{startup} was acquired by {company} in {date}.",
     [("startup", "startup"), ("company", "company"), ("date", "date")],
     [("startup", "company", "acquired_by"), ("startup", "date", "founded_in")]),
    
    # born_in
    ("{person} was born in {city}, {country}.",
     [("person", "person"), ("city", "city"), ("country", "country")],
     [("person", "city", "born_in"), ("city", "country", "part_of")]),
    
    # investor_in
    ("{entrepreneur} invested in {startup}.",
     [("entrepreneur", "entrepreneur"), ("startup", "startup")],
     [("entrepreneur", "startup", "investor_in")]),
    
    # performed_at
    ("{musician} performed at {event}.",
     [("musician", "musician"), ("event", "event")],
     [("musician", "event", "performed_at")]),
    
    # spouse_of
    ("{person} married {person2} in {date}.",
     [("person", "person"), ("person2", "person"), ("date", "date")],
     [("person", "person2", "spouse_of"), ("person", "date", "married_on")]),
    
    # subsidiary_of
    ("{startup} is a subsidiary of {company}.",
     [("startup", "startup"), ("company", "company")],
     [("startup", "company", "subsidiary_of")]),
    
    # research_at
    ("{scientist} conducts research at {research_institute}.",
     [("scientist", "scientist"), ("research_institute", "research_institute")],
     [("scientist", "research_institute", "research_at")]),
    
    # professor_at
    ("{scientist} is a professor at {university}.",
     [("scientist", "scientist"), ("university", "university")],
     [("scientist", "university", "professor_at")]),
    
    # Complex multi-relation sentences
    ("{person} founded {company} in {date}, which developed {product}.",
     [("person", "person"), ("company", "company"), ("date", "date"), ("product", "product")],
     [("person", "company", "founder_of"), ("company", "date", "founded_in"), ("company", "product", "developed")]),
    
    ("{actor} starred in {movie}, directed by {director}.",
     [("actor", "actor"), ("movie", "movie"), ("director", "director")],
     [("actor", "movie", "starred_in"), ("director", "movie", "director_of")]),
    
    ("{musician} won the {award} for {music_album}.",
     [("musician", "musician"), ("award", "award"), ("music_album", "music_album")],
     [("musician", "award", "won"), ("musician", "music_album", "composed_by")]),
    
    ("{company} is headquartered in {city}, {country}.",
     [("company", "company"), ("city", "city"), ("country", "country")],
     [("company", "city", "headquartered_in"), ("city", "country", "part_of")]),
    
    ("{person} graduated from {university} and now works at {company}.",
     [("person", "person"), ("university", "university"), ("company", "company")],
     [("person", "university", "graduated_from"), ("person", "company", "works_at")]),
    
    # AI/Tech specific
    ("{company} released {ai_model} in {date}.",
     [("company", "company"), ("ai_model", "ai_model"), ("date", "date")],
     [("company", "ai_model", "developed"), ("ai_model", "date", "released_in")]),
    
    ("{engineer} created {framework} at {company}.",
     [("engineer", "engineer"), ("framework", "framework"), ("company", "company")],
     [("engineer", "framework", "creator_of"), ("engineer", "company", "works_at")]),
    
    # Sports specific
    ("{athlete} signed with {sports_team} in {date}.",
     [("athlete", "athlete"), ("sports_team", "sports_team"), ("date", "date")],
     [("athlete", "sports_team", "signed_with"), ("athlete", "date", "started_in")]),
    
    ("{athlete} won the {competition} in {date}.",
     [("athlete", "athlete"), ("competition", "competition"), ("date", "date")],
     [("athlete", "competition", "champion_of"), ("competition", "date", "occurred_on")]),
    
    # Media specific
    ("{journalist} works for {media_company}.",
     [("journalist", "journalist"), ("media_company", "media_company")],
     [("journalist", "media_company", "works_at")]),
    
    ("{person} was interviewed by {media_company}.",
     [("person", "person"), ("media_company", "media_company")],
     [("person", "media_company", "featured_in")]),


    ("{person} lives in {city}.", 
     [("person", "person"), ("city", "city")], [("person", "city", "lives_in")]),
    
    ("{person} currently resides in {city}.", 
     [("person", "person"), ("city", "city")], [("person", "city", "lives_in")]),
    
    ("{person} has been staying in {city} for several years.", 
     [("person", "person"), ("city", "city")], [("person", "city", "lives_in")]),
    
    ("The house of {person} is located in {city}.", 
     [("person", "person"), ("city", "city")], [("person", "city", "lives_in")]),

     ("{company} is listed as {stock_symbol} and has a market cap of {money}.", 
     [("company", "company"), ("stock_symbol", "stock_symbol"), ("money", "money")], 
     [("company", "stock_symbol", "listed_as"), ("company", "money", "market_cap")]),
    
    ("{person} owns {percent} of {company} shares.", 
     [("person", "person"), ("percent", "percent"), ("company", "company")], 
     [("person", "percent", "holds_shares_of")]),
    
    ("{company} reported a revenue of {money} in {date}.", 
     [("company", "company"), ("money", "money"), ("date", "date")], 
     [("company", "money", "revenue_of")]),

     ("The official website of {organization} is {url}.", 
     [("organization", "organization"), ("url", "url")], 
     [("organization", "url", "official_website")]),
    
    ("You can contact {person} via email at {email}.", 
     [("person", "person"), ("email", "email")], 
     [("person", "email", "social_media")]),

     ("{medicine} is used to treat {disease}.", 
     [("medicine", "medicine"), ("disease", "disease")], 
     [("medicine", "disease", "treats")]),
    
    ("{person} was diagnosed with {disease}.", 
     [("person", "person"), ("disease", "disease")], 
     [("person", "disease", "diagnosed_with")]),


     # 📑 Formal News / Appositives
    ("{person}, the founder of {company}, announced that {product} was developed in {city}.",
     [("person", "person"), ("company", "company"), ("product", "product"), ("city", "city")],
     [("person", "company", "founder_of"), ("company", "product", "developed"), ("company", "city", "located_in")]),

    # 🤝 Business Partnerships & Supply Chain
    ("{company} acts as a key supplier for {company2}, providing components from its factory in {city}.",
     [("company", "company"), ("company2", "company"), ("city", "city")],
     [("company", "company2", "supplies_to"), ("company", "city", "located_in")]),

    # 🧬 Scientific Discovery (Hard Entities)
    ("The {particle_physics_term} was discovered by {scientist} during research at {research_institute}.",
     [("particle_physics_term", "particle_physics_term"), ("scientist", "scientist"), ("research_institute", "research_institute")],
     [("scientist", "particle_physics_term", "scientific_discovery"), ("scientist", "research_institute", "research_at")]),

    # ⚖️ Legal & Regulatory
    ("Under the {legal_document}, {company} is required to report its annual revenue of {money} to the government.",
     [("legal_document", "legal_document"), ("company", "company"), ("money", "money")],
     [("company", "money", "revenue_of")]),


     # 🏥 Medical & Bio (High Difficulty)
    ("{person} has been diagnosed with {disease} at {research_institute}.", 
     [("person", "person"), ("disease", "disease"), ("research_institute", "research_institute")], 
     [("person", "disease", "diagnosed_with"), ("person", "research_institute", "patient_at")]),
    
    ("{medicine}, which was developed by {company}, is effective against {disease}.", 
     [("medicine", "medicine"), ("company", "company"), ("disease", "disease")], 
     [("company", "medicine", "developed"), ("medicine", "disease", "treats")]),

    # ⚖️ Legal & Governance
    ("The {legal_document} was signed by {person} in {city}, {country}.", 
     [("legal_document", "legal_document"), ("person", "person"), ("city", "city"), ("country", "country")], 
     [("person", "legal_document", "signed_by"), ("city", "country", "part_of")]),

    # 🔬 Science & Space
    ("{scientist} discovered {celestial_object} using the telescope at {research_institute}.", 
     [("scientist", "scientist"), ("celestial_object", "celestial_object"), ("research_institute", "research_institute")], 
     [("scientist", "celestial_object", "discovered_by"), ("scientist", "research_institute", "research_at")]),

    # 💼 Business M&A (Passive Voice)
    ("{startup} was fully acquired for {money} by {company} in {date}.", 
     [("startup", "startup"), ("money", "money"), ("company", "company"), ("date", "date")], 
     [("startup", "company", "acquired_by"), ("company", "money", "paid_for")]),

    # 🏗️ Infrastructure
    ("The construction of {monument} in {city} was led by {engineer}.", 
     [("monument", "monument"), ("city", "city"), ("engineer", "engineer")], 
     [("engineer", "monument", "creator_of"), ("monument", "city", "located_in")]),


     # 🧬 Bio-Tech & Research
    ("The study of {disease} at {research_institute} led to the discovery of {medicine} by {scientist}.",
     [("disease", "disease"), ("research_institute", "research_institute"), ("medicine", "medicine"), ("scientist", "scientist")],
     [("research_institute", "disease", "researches"), ("scientist", "medicine", "inventor_of"), ("medicine", "disease", "treats")]),

    # 🏢 Corporate Governance
    ("Following the resignation of {person}, {person2} was appointed as the new CEO of {company}.",
     [("person", "person"), ("person2", "person"), ("company", "company")],
     [("person", "company", "former_ceo_of"), ("person2", "company", "ceo_of")]),

    # ⚖️ Legal & Regulatory
    ("The {legal_document} signed in {city} mandates that {company} must be a subsidiary of {company2}.",
     [("legal_document", "legal_document"), ("city", "city"), ("company", "company"), ("company2", "company")],
     [("company", "company2", "subsidiary_of"), ("company", "city", "located_in")]),

    # 🛰️ Space & Physics
    ("Observed from {research_institute}, the {celestial_object} was identified as a {particle_physics_term} emitter.",
     [("research_institute", "research_institute"), ("celestial_object", "celestial_object"), ("particle_physics_term", "particle_physics_term")],
     [("research_institute", "celestial_object", "observes")]),



    #all relations covered
    ("{person} is the founder and CEO of {company}, which developed {product} in {city} in {date}.",
     [("person", "person"), ("company", "company"), ("product", "product"), ("city", "city"), ("date", "date")],
     [("person", "company", "founder_of"), ("person", "company", "ceo_of"), ("company", "product", "developed"), ("company", "city", "located_in"), ("company", "date", "founded_in")]),

     ("{person} graduated from {university} in {date} and now works at {company} in {city}.",
     [("person", "person"), ("university", "university"), ("date", "date"), ("company", "company"), ("city", "city")],
     [("person", "university", "graduated_from"), ("person", "date", "graduated_in"), ("person", "company", "works_at"), ("company", "city", "located_in")]),   

     ("{actor} starred in {movie}, directed by {director}, and won the {award} for best performance in {date}.",
     [("actor", "actor"), ("movie", "movie"), ("director", "director"), ("award", "award"), ("date", "date")],
     [("actor", "movie", "starred_in"), ("director", "movie", "director_of"), ("actor", "award", "won"), ("award", "date", "awarded_in")]),

     ("{scientist} discovered {celestial_object} using the telescope at {research_institute} and published the findings in {research_paper}.",
     [("scientist", "scientist"), ("celestial_object", "celestial_object"), ("research_institute", "research_institute"), ("research_paper", "research_paper")],
     [("scientist", "celestial_object", "discovered_by"), ("scientist", "research_institute", "research_at"), ("scientist", "research_paper", "author_of")]),

     ("{company} acquired {startup} for {money} in {date}, with {person} leading the negotiations as CEO.",
     [("company", "company"), ("startup", "startup"), ("money", "money"), ("date", "date"), ("person", "person")],
     [("startup", "company", "acquired_by"), ("company", "money", "paid_for"), ("company", "date", "acquired_in"), ("person", "company", "ceo_of")]),

     ("{person}, the CEO of {company}, announced the development of {product} in {city} in {date}.",
     [("person", "person"), ("company", "company"), ("product", "product"), ("city", "city"), ("date", "date")],
     [("person", "company", "ceo_of"), ("company", "product", "developed"), ("company", "city", "located_in"), ("company", "date", "founded_in")]), 

     ("{person} founded {company} in {date}, which developed {product} and is headquartered in {city}.",
     [("person", "person"), ("company", "company"), ("date", "date"), ("product", "product"), ("city", "city")],
     [("person", "company", "founder_of"), ("company", "date", "founded_in"), ("company", "product", "developed"), ("company", "city", "headquartered_in")]),

     ("{person} graduated from {university} in {date} and now works at {company}, which is headquartered in {city}.",
     [("person", "person"), ("university", "university"), ("date",  "date"), ("company", "company"), ("city", "city")],
     [("person", "university", "graduated_from"), ("person", "date", "graduated_in"), ("person", "company", "works_at"), ("company", "city", "headquartered_in")]),


     ("{actor} starred in {movie}, directed by {director}, and won the {award} for best performance in {date}. The movie was filmed in {city}.",
     [("actor", "actor"), ("movie", "movie"), ("director", "director"), ("award", "award"), ("date", "date"), ("city", "city")],
     [("actor", "movie", "starred_in"), ("director", "movie", "director_of"), ("actor", "award", "won"), ("award", "date", "awarded_in"), ("movie", "city", "filmed_in")]),

     ("{scientist} discovered {celestial_object} using the telescope at {research_institute} and published the findings in {research_paper}. The research institute is located in {city}.",
     [("scientist", "scientist"), ("celestial_object", "celestial_object"), ("research_institute", "research_institute"), ("research_paper", "research_paper"), ("city", "city")],
     [("scientist", "celestial_object", "discovered_by"), ("scientist", "research_institute", "research_at"), ("scientist", "research_paper", "author_of"), ("research_institute", "city", "located_in")]),

     ("{company} acquired {startup} for {money} in {date}, with {person} leading the negotiations as CEO. The startup is based in {city}.",
     [("company", "company"), ("startup", "startup"), ("money", "money"), ("date", "date"), ("person", "person"), ("city", "city")],
     [("startup", "company", "acquired_by"), ("company", "money", "paid_for"), ("company", "date", "acquired_in"), ("person", "company", "ceo_of"), ("startup", "city", "based_in")]),

     ("{person}, the CEO of {company}, announced the development of {product} in {city} in {date}. The product was released globally in {date2}.",
     [("person", "person"), ("company", "company"), ("product", "product"), ("city", "city"), ("date", "date"), ("date2", "date")],
     [("person", "company", "ceo_of"), ("company", "product", "developed"), ("company", "city", "located_in"), ("company", "date", "founded_in"), ("product", "date2", "released_in")]),


    #relation between person and person
    ("{person} is a business partner of {person2}.",
        [("person", "person"), ("person2", "person")],
        [("person", "person2", "partner_with")]),
    
    ("{person} is married to {person2}.",
        [("person", "person"), ("person2", "person")],
        [("person", "person2", "spouse_of")]),

    ("{person} collaborates with {person2} on various projects.",
        [("person", "person"), ("person2", "person")],
        [("person", "person2", "collaborates_with")]),

    ("{person} is the mentor of {person2}.",
        [("person", "person"), ("person2", "person")],
        [("person", "person2", "mentor_of")]),

    ("{person} and {person2} co-founded {company}.",
        [("person", "person"), ("person2", "person"), ("company", "company")],
        [("person", "company", "co_founder_of"), ("person2", "company", "co_founder_of")]),

    ("{person} frequently collaborates with {person2} in their professional endeavors.",
        [("person", "person"), ("person2", "person")],
        [("person", "person2", "collaborates_with")]),





    ("{person} speaks {language} fluently.",
     [("person", "person"), ("language", "language")],
     [("person", "language", "speaks")]),
    
    ("{musician} plays the {instrument} beautifully.",
     [("musician", "musician"), ("instrument", "instrument")],
     [("musician", "instrument", "plays")]),
    
    ("The {animal} is found in the {location}.",
     [("animal", "animal"), ("location", "location")],
     [("animal", "location", "found_in")]),
     
    ("The {animal} mainly eats {food}.",
     [("animal", "animal"), ("food", "food")],
     [("animal", "food", "eats")]),

]

# Chinese Templates
ZH_TEMPLATES = [
    ("{person}於{date}在{city}創立了{company}。",
     [("person", "person"), ("date", "date"), ("city", "city"), ("company", "company")],
     [("person", "company", "founder_of"), ("company", "date", "founded_in"), ("company", "city", "located_in")]),
    
    ("{person}是{company}的執行長。",
     [("person", "person"), ("company", "company")],
     [("person", "company", "ceo_of")]),
    
    ("{company}開發了{product}。",
     [("company", "company"), ("product", "product")],
     [("company", "product", "developed")]),
    
    ("{person}畢業於{university}。",
     [("person", "person"), ("university", "university")],
     [("person", "university", "graduated_from")]),
    
    ("{person}在{company}工作。",
     [("person", "person"), ("company", "company")],
     [("person", "company", "works_at")]),
    
    ("{athlete}效力於{sports_team}。",
     [("athlete", "athlete"), ("sports_team", "sports_team")],
     [("athlete", "sports_team", "plays_for")]),
    
    ("{person}榮獲{award}。",
     [("person", "person"), ("award", "award")],
     [("person", "award", "won")]),
    
    ("{director}執導了{movie}。",
     [("director", "director"), ("movie", "movie")],
     [("director", "movie", "director_of")]),
    
    ("{actor}主演了{movie}。",
     [("actor", "actor"), ("movie", "movie")],
     [("actor", "movie", "starred_in")]),
    
    ("{author}撰寫了{book}。",
     [("author", "author"), ("book", "book")],
     [("author", "book", "author_of")]),
    
    ("{company}收購了{startup}。",
     [("company", "company"), ("startup", "startup")],
     [("startup", "company", "acquired_by")]),
    
    ("{person}出生於{city}。",
     [("person", "person"), ("city", "city")],
     [("person", "city", "born_in")]),
    
    ("{company}總部位於{city}。",
     [("company", "company"), ("city", "city")],
     [("company", "city", "headquartered_in")]),
    
    ("{scientist}是{university}的教授。",
     [("scientist", "scientist"), ("university", "university")],
     [("scientist", "university", "professor_at")]),
    
    ("{musician}演唱了{music_album}。",
     [("musician", "musician"), ("music_album", "music_album")],
     [("musician", "music_album", "composed_by")]),

    ("{person}{company}的合作夥伴。", 
     [("person", "person"), ("company", "company")], 
     [("person", "company", "partner_with")]),


     ("{person}居住在{city}。", 
     [("person", "person"), ("city", "city")], [("person", "city", "lives_in")]),
    
    ("{person}目前定居於{city}。", 
     [("person", "person"), ("city", "city")], [("person", "city", "lives_in")]),

     # 🏢 企業併購 (M&A)
    ("總部位於{city}的{company}宣佈以{money}的價格收購了初創公司{startup}。",
     [("city", "city"), ("company", "company"), ("money", "money"), ("startup", "startup")],
     [("company", "city", "headquartered_in"), ("company", "startup", "acquired_by")]),

    # 🔬 科學研究
    ("{scientist}在{university}實驗室工作期間，成功研發了新技術{invention}，並獲得了{award}。",
     [("scientist", "scientist"), ("university", "university"), ("invention", "invention"), ("award", "award")],
     [("scientist", "university", "research_at"), ("scientist", "invention", "inventor_of"), ("scientist", "award", "won")]),


    # 🎓 Academic & Career
    ("{person}在{university}獲得了{academic_field}學位，隨後加入{company}。", 
     [("person", "person"), ("university", "university"), ("academic_field", "academic_field"), ("company", "company")], 
     [("person", "university", "graduated_from"), ("person", "academic_field", "studied_at"), ("person", "company", "works_at")]),

    # 🎬 Media & Entertainment
    ("由{director}執導、{actor}主演的電影{movie}在{city}舉行了首映禮。", 
     [("director", "director"), ("actor", "actor"), ("movie", "movie"), ("city", "city")], 
     [("director", "movie", "director_of"), ("actor", "movie", "starred_in"), ("movie", "city", "premiered_in")]),

    # 🏙️ Geographic & Corporate
    ("{company}將其全球總部從{city}搬遷到了{city2}。", 
     [("company", "company"), ("city", "city"), ("city2", "city")], 
     [("company", "city", "formerly_at"), ("company", "city2", "headquartered_in")]),

    # 🏗️ 基礎設施與工程
    ("由 {engineer} 負責設計並位於 {city} 的 {monument} 於 {date} 正式完工。",
     [("engineer", "engineer"), ("city", "city"), ("monument", "monument"), ("date", "date")],
     [("engineer", "monument", "creator_of"), ("monument", "city", "located_in")]),

    # 💻 科技研發
    ("{company} 在 {city} 的研發中心成功開發了名為 {ai_model} 的人工智慧系統。",
     [("company", "company"), ("city", "city"), ("ai_model", "ai_model")],
     [("company", "ai_model", "developed"), ("company", "city", "located_in")]),
     

]

# Japanese Templates
JA_TEMPLATES = [
    ("{person}は{date}に{city}で{company}を設立した。",
     [("person", "person"), ("date", "date"), ("city", "city"), ("company", "company")],
     [("person", "company", "founder_of"), ("company", "date", "founded_in"), ("company", "city", "located_in")]),
    
    ("{person}は{company}のCEOである。",
     [("person", "person"), ("company", "company")],
     [("person", "company", "ceo_of")]),
    
    ("{company}は{product}を開発した。",
     [("company", "company"), ("product", "product")],
     [("company", "product", "developed")]),
    
    ("{person}は{university}を卒業した。",
     [("person", "person"), ("university", "university")],
     [("person", "university", "graduated_from")]),
    
    ("{person}は{company}で働いている。",
     [("person", "person"), ("company", "company")],
     [("person", "company", "works_at")]),
    
    ("{athlete}は{sports_team}でプレーしている。",
     [("athlete", "athlete"), ("sports_team", "sports_team")],
     [("athlete", "sports_team", "plays_for")]),
    
    ("{person}は{award}を受賞した。",
     [("person", "person"), ("award", "award")],
     [("person", "award", "won")]),
    
    ("{director}は{movie}を監督した。",
     [("director", "director"), ("movie", "movie")],
     [("director", "movie", "director_of")]),
    
    ("{actor}は{movie}に出演した。",
     [("actor", "actor"), ("movie", "movie")],
     [("actor", "movie", "starred_in")]),
    
    ("{author}は{book}を執筆した。",
     [("author", "author"), ("book", "book")],
     [("author", "book", "author_of")]),
    
    ("{company}は{startup}を買収した。",
     [("company", "company"), ("startup", "startup")],
     [("startup", "company", "acquired_by")]),
    
    ("{person}は{city}で生まれた。",
     [("person", "person"), ("city", "city")],
     [("person", "city", "born_in")]),
    
    ("{company}の本社は{city}にある。",
     [("company", "company"), ("city", "city")],
     [("company", "city", "headquartered_in")]),
    
    ("{scientist}は{university}の教授である。",
     [("scientist", "scientist"), ("university", "university")],
     [("scientist", "university", "professor_at")]),

     ("{person}は{city}に住んでいます。", 
     [("person", "person"), ("city", "city")], [("person", "city", "lives_in")]),
    
    ("{person}の自宅は{city}にあります。", 
     [("person", "person"), ("city", "city")], [("person", "city", "lives_in")]),


     ("{person}氏は、{company}の創業者であり、現在は{university}で客員教授を務めている。", 
     [("person", "person"), ("company", "company"), ("university", "university")], 
     [("person", "company", "founder_of"), ("person", "university", "professor_at")]),
]

# Korean Templates
KO_TEMPLATES = [
    ("{person}은(는) {date} {city}에서 {company}을(를) 설립했다.",
     [("person", "person"), ("date", "date"), ("city", "city"), ("company", "company")],
     [("person", "company", "founder_of"), ("company", "date", "founded_in"), ("company", "city", "located_in")]),
    
    ("{person}은(는) {company}의 CEO이다.",
     [("person", "person"), ("company", "company")],
     [("person", "company", "ceo_of")]),
    
    ("{company}은(는) {product}을(를) 개발했다.",
     [("company", "company"), ("product", "product")],
     [("company", "product", "developed")]),
    
    ("{person}은(는) {university}를 졸업했다.",
     [("person", "person"), ("university", "university")],
     [("person", "university", "graduated_from")]),
    
    ("{person}은(는) {company}에서 일한다.",
     [("person", "person"), ("company", "company")],
     [("person", "company", "works_at")]),
    
    ("{athlete}은(는) {sports_team}에서 뛴다.",
     [("athlete", "athlete"), ("sports_team", "sports_team")],
     [("athlete", "sports_team", "plays_for")]),
    
    ("{person}은(는) {award}을(를) 수상했다.",
     [("person", "person"), ("award", "award")],
     [("person", "award", "won")]),
    
    ("{director}은(는) {movie}를 감독했다.",
     [("director", "director"), ("movie", "movie")],
     [("director", "movie", "director_of")]),
    
    ("{actor}은(는) {movie}에 출연했다.",
     [("actor", "actor"), ("movie", "movie")],
     [("actor", "movie", "starred_in")]),
    
    ("{company}은(는) {startup}을(를) 인수했다.",
     [("company", "company"), ("startup", "startup")],
     [("startup", "company", "acquired_by")]),
    
    ("{person}은(는) {city}에서 태어났다.",
     [("person", "person"), ("city", "city")],
     [("person", "city", "born_in")]),
    
    ("{company}의 본사는 {city}에 있다.",
     [("company", "company"), ("city", "city")],
     [("company", "city", "headquartered_in")]),

     ("{person}은(는) {city}에 거주하고 있다.", 
     [("person", "person"), ("city", "city")], [("person", "city", "lives_in")]),
    
    ("{person}의 집은 {city}에 있다.", 
     [("person", "person"), ("city", "city")], [("person", "city", "lives_in")]),


     ("{company}은(는) {city}에 위치한 {research_institute}와(과) 전략적 파트너십을 체결했다.", 
     [("company", "company"), ("city", "city"), ("research_institute", "research_institute")], 
     [("company", "research_institute", "partner_with"), ("research_institute", "city", "located_in")]),
]

# Thai Templates
TH_TEMPLATES = [
    ("{person} ก่อตั้ง {company} ในปี {date} ที่{city}",
     [("person", "person"), ("company", "company"), ("date", "date"), ("city", "city")],
     [("person", "company", "founder_of"), ("company", "date", "founded_in"), ("company", "city", "located_in")]),
    
    ("{person} เป็นซีอีโอของ {company}",
     [("person", "person"), ("company", "company")],
     [("person", "company", "ceo_of")]),
    
    ("{company} พัฒนา {product}",
     [("company", "company"), ("product", "product")],
     [("company", "product", "developed")]),
    
    ("{person} จบการศึกษาจาก {university}",
     [("person", "person"), ("university", "university")],
     [("person", "university", "graduated_from")]),
    
    ("{person} ทำงานที่ {company}",
     [("person", "person"), ("company", "company")],
     [("person", "company", "works_at")]),
    
    ("{athlete} เล่นให้ {sports_team}",
     [("athlete", "athlete"), ("sports_team", "sports_team")],
     [("athlete", "sports_team", "plays_for")]),
    
    ("{person} ได้รับรางวัล {award}",
     [("person", "person"), ("award", "award")],
     [("person", "award", "won")]),
    
    ("{director} กำกับ {movie}",
     [("director", "director"), ("movie", "movie")],
     [("director", "movie", "director_of")]),
    
    ("{actor} แสดงใน {movie}",
     [("actor", "actor"), ("movie", "movie")],
     [("actor", "movie", "starred_in")]),
    
    ("{person} เกิดที่ {city}",
     [("person", "person"), ("city", "city")],
     [("person", "city", "born_in")]),
    
    ("{company} มีสำนักงานใหญ่ที่{city}",
     [("company", "company"), ("city", "city")],
     [("company", "city", "headquartered_in")]),

     ("{person} พักอาศัยอยู่ที่{city}", 
     [("person", "person"), ("city", "city")], [("person", "city", "lives_in")]),
    
    ("บ้านของ {person} อยู่ที่{city}", 
     [("person", "person"), ("city", "city")], [("person", "city", "lives_in")]),
    
    ("{person} ใช้ชีวิตส่วนใหญ่อยู่ใน{city}", 
     [("person", "person"), ("city", "city")], [("person", "city", "lives_in")]),

     ("เว็บไซต์ทางการของ {organization} คือ {url}", 
     [("organization", "organization"), ("url", "url")], 
     [("organization", "url", "official_website")]),

     ("{company} จดทะเบียนในตลาดหลักทรัพย์ด้วยชื่อ {stock_symbol} และมีมูลค่าบริษัท {money}", 
     [("company", "company"), ("stock_symbol", "stock_symbol"), ("money", "money")], 
     [("company", "stock_symbol", "listed_as"), ("company", "money", "market_cap")]),
    
    ("{person} ถือหุ้นจำนวน {percent} ใน {company}", 
     [("person", "person"), ("percent", "percent"), ("company", "company")], 
     [("person", "percent", "holds_shares_of")]),   

     # 📰 รายงานข่าวเศรษฐกิจ
    ("{company} ยักษ์ใหญ่จาก{country} มีแผนขยายฐานการผลิต{product}ไปยัง{city}ภายในปี {date}",
     [("company", "company"), ("country", "country"), ("product", "product"), ("city", "city"), ("date", "date")],
     [("company", "country", "based_in"), ("company", "product", "developed"), ("company", "city", "located_in")]),

    # 🎓 ประวัติบุคคลสำคัญ
    ("หลังจากที่{person}สำเร็จการศึกษาด้าน{academic_field}จาก{university} เขาก็ได้เริ่มทำงานที่{company}",
     [("person", "person"), ("academic_field", "academic_field"), ("university", "university"), ("company", "company")],
     [("person", "university", "graduated_from"), ("person", "academic_field", "studied_at"), ("person", "company", "works_at")]),

    # 💊 การแพทย์และสาธารณสุข
    ("แพทย์วินิจฉัยว่า{person}ป่วยเป็นโรค{disease} และแนะนำให้ใช้ยา{medicine}เพื่อรักษาอาการบริเวณ{organ}",
     [("person", "person"), ("disease", "disease"), ("medicine", "medicine"), ("organ", "organ")],
     [("person", "disease", "diagnosed_with"), ("medicine", "disease", "treats"), ("disease", "organ", "affects")]),


    # 📰 ข่าวธุรกิจและการลงทุน
    ("{company} ภายใต้การนำของ {person} ได้ประกาศควบรวมกิจการกับ {startup} ที่มูลค่า {money}", 
     [("company", "company"), ("person", "person"), ("startup", "startup"), ("money", "money")], 
     [("person", "company", "ceo_of"), ("company", "startup", "acquired_by")]),

    # 🏛️ ประวัติศาสตร์และวัฒนธรรม
    ("{monument} ถูกสร้างขึ้นในสมัยของ {person} เพื่อเป็นสัญลักษณ์ของ {city}", 
     [("monument", "monument"), ("person", "person"), ("city", "city")], 
     [("person", "monument", "creator_of"), ("monument", "city", "located_in")]),

    # 🧪 วิทยาศาสตร์และนวัตกรรม
    ("{scientist} จาก {university} ประสบความสำเร็จในการค้นพบ {invention} ซึ่งจะช่วยรักษา {disease}", 
     [("scientist", "scientist"), ("university", "university"), ("invention", "invention"), ("disease", "disease")], 
     [("scientist", "university", "research_at"), ("scientist", "invention", "inventor_of"), ("invention", "disease", "treats")]),
    

    # 📰 รายงานข่าวและการเมือง
    ("ภายใต้ข้อตกลง {legal_document} ระบุว่า {company} จะเข้าซื้อกิจการ {startup} ในมูลค่า {money}",
     [("legal_document", "legal_document"), ("company", "company"), ("startup", "startup"), ("money", "money")],
     [("company", "startup", "acquired_by")]),

    # 💊 การแพทย์และสาธารณสุข
    ("ผลการทดสอบ {medicine} ใน {research_institute} พบว่าสามารถยับยั้ง {disease} ที่ส่งผลต่อ {organ} ได้",
     [("medicine", "medicine"), ("research_institute", "research_institute"), ("disease", "disease"), ("organ", "organ")],
     [("medicine", "disease", "treats"), ("disease", "organ", "affects")]),

    # 🎓 การศึกษาและบุคคล
    ("{person} ผู้เชี่ยวชาญด้าน {academic_field} จาก {university} ได้รับเลือกให้เป็น {title} ประจำปี {date}",
     [("person", "person"), ("academic_field", "academic_field"), ("university", "university"), ("title", "title"), ("date", "date")],
     [("person", "university", "graduated_from"), ("person", "academic_field", "specialist_in")]),

    ("{person} พูดภาษา{language}ได้อย่างคล่องแคล่ว",
     [("person", "person"), ("language", "language")],
     [("person", "language", "speaks")]),
    
    ("{musician} เล่น{instrument}ได้อย่างไพเราะ",
     [("musician", "musician"), ("instrument", "instrument")],
     [("musician", "instrument", "plays")]),

    ("{animal} มักพบได้ใน{location}",
     [("animal", "animal"), ("location", "location")],
     [("animal", "location", "found_in")]),
     
    ("อาหารหลักของ{animal} คือ {food}",
     [("animal", "animal"), ("food", "food")],
     [("animal", "food", "eats")]),
]

# (Template, Entity_List, Relation_List)
EN_COMPLEX_TEMPLATES = [
    # Relations: founder_of, founded_in, located_in, developed
    ("{person}, who founded {company} in {date} at {city}, recently announced that they developed {product}.",
     [("person", "person"), ("company", "company"), ("date", "date"), ("city", "city"), ("product", "product")],
     [("person", "company", "founder_of"), 
      ("company", "date", "founded_in"), 
      ("company", "city", "located_in"), 
      ("company", "product", "developed")]),

    # Relations: ceo_of, headquartered_in, acquired_by, subsidiary_of
    ("As the CEO of {company} based in {city}, {person} oversaw the acquisition of {startup} which is now a subsidiary of {company}.",
     [("person", "person"), ("company", "company"), ("city", "city"), ("startup", "startup")],
     [("person", "company", "ceo_of"), 
      ("company", "city", "headquartered_in"), 
      ("startup", "company", "acquired_by"), 
      ("startup", "company", "subsidiary_of")]),
]
EN_COMPLEX_TEMPLATES += [
    # เพิ่มส่วนขยายแบบทางการ (News style)
    ("{person}, the renowned {title} of {company}, announced from {city} that their latest innovation, {product}, will launch in {date}.",
     [("person", "person"), ("title", "title"), ("company", "company"), ("city", "city"), ("product", "product"), ("date", "date")],
     [("person", "company", "ceo_of"), ("person", "title", "achieved"), ("company", "city", "located_in"), ("company", "product", "developed")]),
    
    # การใช้สรรพนามเชื่อมโยง (Anaphora Reference)
    ("After graduating from {university}, {person} joined {company} in {city}; shortly after, they became the {title} of the firm.",
     [("university", "university"), ("person", "person"), ("company", "company"), ("city", "city"), ("title", "title")],
     [("person", "university", "graduated_from"), ("person", "company", "works_at"), ("person", "title", "achieved"), ("company", "city", "located_in")]),
]
EN_COMPLEX_TEMPLATES += [
    # A -> subsidiary_of -> B AND A -> developed -> C
    ("{company}, a subsidiary of {company2}, officially released its new {product} in {city}.",
     [("company", "company"), ("company2", "company"), ("product", "product"), ("city", "city")],
     [("company", "company2", "subsidiary_of"), 
      ("company", "product", "developed"),
      ("company", "city", "located_in")]),
]

ZH_COMPLEX_TEMPLATES = [
    # Relations: ceo_of, headquartered_in, graduated_from, studied_at
    ("總部位於{city}的{company}執行長{person}，曾就讀於{university}並在那裡獲得了學位。",
     [("city", "city"), ("company", "company"), ("person", "person"), ("university", "university")],
     [("company", "city", "headquartered_in"), 
      ("person", "company", "ceo_of"), 
      ("person", "university", "graduated_from"), 
      ("person", "university", "studied_at")]),

    # Relations: founder_of, developed, released_in
    ("{person}在{city}創立了{company}後，隨即推出了在{date}開發的{product}。",
     [("person", "person"), ("city", "city"), ("company", "company"), ("date", "date"), ("product", "product")],
     [("person", "company", "founder_of"), 
      ("company", "city", "located_in"), 
      ("company", "product", "developed"), 
      ("product", "date", "released_in")]),
]

TH_COMPLEX_TEMPLATES = [
    # Relations: ceo_of, headquartered_in, acquired_by, developed
    ("{person} ซีอีโอของ {company} ซึ่งมีสำนักงานใหญ่ที่{city} ได้เข้าซื้อกิจการ {startup} ผู้พัฒนา {product}",
     [("person", "person"), ("company", "company"), ("city", "city"), ("startup", "startup"), ("product", "product")],
     [("person", "company", "ceo_of"), 
      ("company", "city", "headquartered_in"), 
      ("startup", "company", "acquired_by"), 
      ("startup", "product", "developed")]),

    # Relations: founder_of, founded_in, graduated_from, lives_in
    ("{person} ผู้ก่อตั้ง {company} เมื่อปี {date} เป็นศิษย์เก่าจาก {university} และปัจจุบันอาศัยอยู่ที่{city}",
     [("person", "person"), ("company", "company"), ("date", "date"), ("university", "university"), ("city", "city")],
     [("person", "company", "founder_of"), 
      ("company", "date", "founded_in"), 
      ("person", "university", "graduated_from"), 
      ("person", "city", "lives_in")]),
]
TH_COMPLEX_TEMPLATES += [
    # แบบทางการ (News Style)
    ("รายงานจาก{city}ระบุว่า {person} ในฐานะ{title}ของ{company} ได้เปิดตัว {product} อย่างเป็นทางการเมื่อ{date}",
     [("city", "city"), ("person", "person"), ("title", "title"), ("company", "company"), ("product", "product"), ("date", "date")],
     [("person", "company", "ceo_of"), ("person", "title", "achieved"), ("company", "city", "located_in"), ("company", "product", "developed")]),
    
    # แบบความสัมพันธ์ซ้อน (Nested Relations)
    ("{company} ซึ่งเป็นบริษัทในเครือของ {company2} และมีสำนักงานใหญ่ที่{city} ได้แต่งตั้ง {person} เป็นซีอีโอคนใหม่",
     [("company", "company"), ("company2", "company"), ("city", "city"), ("person", "person")],
     [("company", "company2", "subsidiary_of"), ("company", "city", "headquartered_in"), ("person", "company", "ceo_of")]),
]

# ============================================================================
# NOISE FUNCTIONS
# ============================================================================


def apply_coreference_logic(sample_data):
    text = sample_data["text"]
    entities = sample_data["entities"]
    relations = sample_data["relations"]
    
    # พจนานุกรมสรรพนามแยกตามภาษา
    pronoun_map = {
        "en": {"person": "He", "organization": "It", "company": "It", "default": "It", "suffix": " is leading the field."},
        "th": {"person": "เขา", "organization": "มัน", "company": "องค์กรนี้", "default": "สิ่งนี้", "suffix": "กำลังเป็นผู้นำในอุตสาหกรรม"},
        "zh": {"person": "他", "organization": "它", "company": "該公司", "default": "它", "suffix": "目前在行業中處於領先地位"},
        "ja": {"person": "彼", "organization": "それ", "company": "同社", "default": "それ", "suffix": "は現在業界をリードしています"},
        "ko": {"person": "그", "organization": "그것", "company": "이 회사는", "default": "그것", "suffix": " 현재 업계를 선도하고 있습니다"}
    }

    if entities and random.random() < 0.3:
        # ตรวจสอบภาษาของ Text
        lang = "en"
        if any('\u0e00' <= c <= '\u0e7f' for c in text): lang = "th"
        elif any('\u4e00' <= c <= '\u9fff' for c in text): lang = "zh"
        elif any('\u3040' <= c <= '\u30ff' for c in text): lang = "ja"
        elif any('\uac00' <= c <= '\ud7af' for c in text): lang = "ko"

        target_ent = random.choice(entities)
        label = target_ent["label"]
        
        # เลือกสรรพนาม (Fallback ไปที่ default)
        category = "person" if label in ["person", "politician", "scientist", "actor"] else ("company" if label in ["company", "startup"] else "default")
        pronoun = pronoun_map[lang].get(category, pronoun_map[lang]["default"])
        full_new_sentence = f" {pronoun}{pronoun_map[lang]['suffix']}"
        
        # บันทึกตำแหน่งก่อนเพิ่มประโยคใหม่
        old_len = len(text)
        text += full_new_sentence
        
        # คำนวณตำแหน่งสรรพนามใน Text ใหม่
        pronoun_start = old_len + 1 # +1 สำหรับช่องว่าง
        
        description = target_ent.get("description", "")
        
        entities.append({
            "start": pronoun_start,
            "end": pronoun_start + len(pronoun),
            "label": label, # ใช้ Label เดียวกับตัวหลัก
            "text": pronoun,
            "description": description
        })
        
    return text, entities, relations


def add_hard_negatives(sample_data, entities_dict):
    text = sample_data["text"]
    entities = sample_data["entities"]
    
    # โอกาสเกิด 40%
    if random.random() < 0.4:
        # สุ่ม Entity หลอกที่ไม่อยู่ในประโยคเดิม
        fake_type = random.choice(["person", "company", "city"])
        fake_val = get_entity(entities_dict, fake_type)
        
        # ตรวจสอบภาษาเพื่อเลือก Filler phrases
        lang = "en"
        if any('\u0e00' <= c <= '\u0e7f' for c in text): lang = "th"
        elif any('\u4e00' <= c <= '\u9fff' for c in text): lang = "zh"

        noise_templates = {
            "en": [f"Unlike {fake_val}, ", f"While {fake_val} was absent, ", f". Note: {fake_val} ignored this."],
            "th": [f"ต่างจาก {fake_val} ", f"ในขณะที่ {fake_val} ไม่ได้เข้าร่วม ", f". หมายเหตุ: {fake_val} ไม่เกี่ยวข้อง"],
            "zh": [f"與 {fake_val} 不同，", f"在 {fake_val} 缺席的情況下，", f"。註：{fake_val} 未參與"]
        }
        
        chosen_noise = random.choice(noise_templates.get(lang, noise_templates["en"]))
        
        # กรณี Noise อยู่ข้างหน้า (ต้องแก้ Offset ของ Entity เก่าทั้งหมด)
        if chosen_noise.startswith(("Unlike", "While", "ต่างจาก", "ในขณะที่", "與", "在")):
            offset = len(chosen_noise)
            text = chosen_noise + text
            for ent in entities:
                ent["start"] += offset
                ent["end"] += offset
            
            # เพิ่ม Entity หลอกลงใน Metadata (แต่ไม่เพิ่มใน Relations!)
            description = get_label_description(fake_type, label_type="entity")
            entities.append({
                "start": chosen_noise.find(fake_val),
                "end": chosen_noise.find(fake_val) + len(fake_val),
                "label": fake_type,
                "text": fake_val,
                "description": description
            })
        else:
            # กรณี Noise อยู่ข้างหลัง (ไม่ต้องแก้ Offset)
            start_pos = len(text) + chosen_noise.find(fake_val)
            text = text + chosen_noise
            description = get_label_description(fake_type, label_type="entity")
            entities.append({
                "start": start_pos,
                "end": start_pos + len(fake_val),
                "label": fake_type,
                "text": fake_val,
                "description": description
            })
            
    return text, entities

def apply_contextual_padding(text: str) -> Tuple[str, int]:
    """คืนค่า (text_ใหม่, offset) โดยรองรับ 5 ภาษาหลัก"""
    
    # เช็คภาษาจากตัวอักษรในประโยค
    is_thai = any('\u0e00' <= char <= '\u0e7f' for char in text)
    is_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    is_japanese = any('\u3040' <= char <= '\u30ff' for char in text)
    is_korean = any('\uac00' <= char <= '\ud7af' for char in text)

    # คลังคำเกริ่นนำแยกตามภาษา
    multilingual_prefixes = {
        "en": ["In a recent development, ", "According to reports, ", "Sources indicate that "],
        "th": ["มีรายงานระบุว่า ", "ข้อมูลล่าสุดเปิดเผยว่า ", "ตามรายงานจากแหล่งข่าว "],
        "zh": ["據近期消息指出，", "根據官方發佈的聲明，", "相關報導顯示，"],
        "ja": ["最新の報道によると、", "関係者からの情報では、", "公式発表によれば、"],
        "ko": ["최근 보도에 따르면, ", "공식 발표에 따르면, ", "업계 관계자에 따르면, "]
    }
    
    multilingual_suffixes = {
        "en": [", causing market shifts.", " according to sources.", " for the upcoming fiscal year."],
        "th": [" ซึ่งส่งผลกระทบต่อตลาดโลก", " ตามข้อมูลจากแหล่งข่าวใกล้ชิด", " โดยคาดว่าจะเห็นผลในปีนี้"],
        "zh": ["，這引起了市場的劇烈波動。", "，據相關人士透露。", "，預計將在下個季度完成。"],
        "ja": ["、これにより市場に大きな影響が出ています。", "、関係者が明らかにしました。", "、来期までに完了する見込みです。"],
        "ko": [", 이는 시장에 큰 영향을 미치고 있습니다.", ", 관계자의 설명입니다.", ", 내년까지 완료될 예정입니다."]
    }

    # เลือกภาษาหลัก
    lang = "en"
    if is_thai: lang = "th"
    elif is_chinese: lang = "zh"
    elif is_japanese: lang = "ja"
    elif is_korean: lang = "ko"

    offset = 0
    new_text = text

    # 1. จัดการ Prefix (บวก Offset)
    if random.random() < 0.4:
        prefix = random.choice(multilingual_prefixes[lang])
        offset = len(prefix)
        
        # สำหรับภาษาอังกฤษ (EN) ให้ปรับตัวแรกเป็นตัวเล็ก ถ้าไม่ใช่ตัวพิมพ์ใหญ่ (เช่น ชื่อคน)
        if lang == "en" and text[0].islower():
            new_text = prefix + text[0].lower() + text[1:]
        else:
            new_text = prefix + text
    
    # 2. จัดการ Suffix (ไม่ต้องแก้ Offset)
    if random.random() < 0.3:
        suffix = random.choice(multilingual_suffixes[lang])
        new_text = new_text.rstrip(' .。') + suffix
        
    return new_text, offset


def apply_typo(text: str, probability=0.1) -> str:
    """จำลองการพิมพ์ผิด (Typos)"""
    if random.random() > probability or len(text) < 4:
        return text
    chars = list(text)
    idx = random.randint(0, len(chars) - 2)
    # สลับตำแหน่งตัวอักษร
    chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    return "".join(chars)

def apply_case_variation(text: str, probability=0.2) -> str:
    """จำลองความไม่สม่ำเสมอของตัวพิมพ์ (Case Variation)"""
    if random.random() > probability:
        return text
    case_types = [
        lambda t: t.upper(),      # ELON MUSK
        lambda t: t.lower(),      # elon musk
        lambda t: t.swapcase(),   # eLON mUSK
    ]
    return random.choice(case_types)(text)

def apply_incomplete_entity(text: str, probability=0.1) -> str:
    """จำลองชื่อที่ไม่สมบูรณ์ (Incomplete Entities)"""
    if random.random() > probability or " " not in text:
        return text
    parts = text.split()
    # เก็บไว้แค่บางส่วน เช่น "Elon Musk" -> "Musk"
    return random.choice(parts)

# รายชื่อคำกริยา/คำรอบข้างที่ใช้แทนกันได้ (Synonym Replacement)
SYNONYMS = {
    "founded": ["started", "created", "established", "launched", "set up"],
    "works at": ["is employed by", "is part of", "serves at", "is a member of"],
    "won": ["received", "was awarded", "took home", "secured"],
}

def apply_synonym(template_text: str) -> str:
    """สุ่มเปลี่ยนคำกริยาใน Template"""
    for key, subs in SYNONYMS.items():
        if key in template_text:
            template_text = template_text.replace(key, random.choice(subs))
    return template_text

def apply_chinese_noise(text: str, probability=0.1) -> str:
    """จำลองความผิดพลาดในภาษาจีน"""
    if not any('\u4e00' <= char <= '\u9fff' for char in text) or random.random() > probability:
        return text
    
    chars = list(text)
    noise_type = random.random()
    
    # 1. Homophone Errors (ตัวอักษรที่เสียงพ้องแต่เขียนผิด - พบบ่อยมากในแชท)
    # ในที่นี้จำลองโดยการเปลี่ยนตัวอักษรที่คนมักพิมพ์ผิด
    if noise_type < 0.4 and len(chars) > 2:
        idx = random.randint(0, len(chars) - 1)
        # ตัวอย่าง: เปลี่ยน 的 เป็น 地 หรือ 得 หรือตัวที่หน้าตาคล้ายกัน
        confusing_chars = {'的': '得', '在': '再', '做': '作', '妳': '你', '公': '工'}
        if chars[idx] in confusing_chars:
            chars[idx] = confusing_chars[chars[idx]]

    # 2. Traditional vs Simplified Mix (จำลองการใช้ปนกัน)
    elif noise_type < 0.7:
        # ตัวอย่างง่ายๆ: เปลี่ยนตัวย่อเป็นตัวเต็มบางตัว
        sim_to_trad = {'台': '臺', '国': '國', '学': '學', '会': '會', '发': '發'}
        idx = random.randint(0, len(chars) - 1)
        if chars[idx] in sim_to_trad:
            chars[idx] = sim_to_trad[chars[idx]]

    # 3. Punctuation/Space Noise (ภาษาจีนมักไม่มีเว้นวรรค แต่บางคนชอบใส่)
    else:
        idx = random.randint(1, len(chars) - 1)
        chars.insert(idx, " ") # ใส่เว้นวรรคผิดตำแหน่งเพื่อหลอก Segmentation
        
    return "".join(chars)

def apply_chinese_incomplete(text: str, probability=0.15) -> str:
    """จำลองชื่อเรียกย่อในภาษาจีน (Incomplete/Shortened names)"""
    if not any('\u4e00' <= char <= '\u9fff' for char in text) or random.random() > probability:
        return text
    
    # ชื่อคนจีน 3 ตัว มักถูกเรียกเหลือ 2 ตัว (เช่น "張忠謀" -> "張大" หรือตัดแซ่ออก)
    if len(text) >= 3:
        return text[1:] # ตัดแซ่ (Surname) ออก เหลือแต่ชื่อ
    return text

def apply_thai_keyboard_shift(text: str, probability=0.05) -> str:
    """จำลองการลืมเปลี่ยนภาษา (เช่น พิมพ์ 'hello' เป็น 'เ้สสว') หรือพิมพ์ผิดปุ่มใกล้เคียง"""
    if not text or random.random() > probability:
        return text
    
    # ตัวอย่าง map แป้นพิมพ์ (Simplified) - จำลองแค่บางตัวที่พบบ่อย
    kb_map = {'ก': 'ด', 'ด': 'ก', 'า': 'ส', 'ส': 'า', 'เ': 'แ', 'แ': 'เ', 'ิ': 'ี', 'ี': 'ิ'}
    chars = list(text)
    idx = random.randint(0, len(chars) - 1)
    if chars[idx] in kb_map:
        chars[idx] = kb_map[chars[idx]]
    return "".join(chars)

def apply_thai_vowel_noise(text: str, probability=0.1) -> str:
    """จำลองการพิมพ์สระ/วรรณยุกต์ผิด หรือวางตำแหน่งผิด (พบบ่อยใน Social Media)"""
    if not any('\u0e00' <= char <= '\u0e7f' for char in text) or random.random() > probability:
        return text
    
    chars = list(text)
    noise_type = random.random()
    
    # 1. การใช้สระเสียงสั้น/ยาว สลับกัน (เช่น 'คะ' -> 'ค่ะ', 'นะ' -> 'น้า')
    if noise_type < 0.5:
        vowel_swaps = {'ะ': 'คะ', 'า': 'ะ', 'ิ': 'ี', 'ุ': 'ู'}
        # สุ่มเปลี่ยนสระท้ายคำ
        for i in range(len(chars)-1, -1, -1):
            if chars[i] in vowel_swaps:
                chars[i] = vowel_swaps[chars[i]]
                break
                
    # 2. จำลองการพิมพ์ "นะคร้าบบบ" (ตัวอักษรซ้ำ)
    else:
        if len(chars) > 0:
            idx = len(chars) - 1
            chars.append(chars[idx] * random.randint(1, 3))
            
    return "".join(chars)

def apply_thai_slang_shorten(text: str, probability=0.1) -> str:
    """จำลองการตัดคำย่อในภาษาไทย (เช่น 'มหาวิทยาลัย' -> 'ม.', 'จังหวัด' -> 'จ.')"""
    short_map = {
        "มหาวิทยาลัย": "ม.",
        "จังหวัด": "จ.",
        "บริษัท": "บจก.",
        "ถนน": "ถ.",
        "ตำบล": "ต.",
        "อำเภอ": "อ."
    }
    for long_form, short_form in short_map.items():
        if long_form in text and random.random() < probability:
            return text.replace(long_form, short_form)
    return text


def apply_indirect_reference(text: str, entity_text: str, entity_type: str, probability=0.2) -> str:
    """จำลอง Anaphora Resolution: แทนที่การเรียกชื่อซ้ำด้วยคำสรรพนาม"""
    if random.random() > probability:
        return text
    
    pronouns = {
        "person": ["he", "she", "this individual", "the person"],
        "company": ["it", "the company", "the firm", "this organization"],
        "location": ["there", "the city", "this region"]
    }
    
    label = "person" if entity_type in ["person", "politician", "scientist", "engineer"] else "company"
    if entity_type in ["city", "country", "location"]: label = "location"
    
    replacement = random.choice(pronouns.get(label, ["it"]))
    
    # แทนที่เฉพาะจุดที่สองที่เจอชื่อเดิม (ถ้ามี)
    parts = text.split(entity_text)
    if len(parts) > 2:
        return entity_text.join(parts[:-1]) + replacement + parts[-1]
    return text

def apply_semantic_reversal(template: str, entity_defs: List, relation_defs: List) -> Tuple[str, List, List]:
    """สลับโครงสร้างประโยค (Semantic Reversal) เช่น Active เป็น Passive
    
    🔥 FIX: Language-Aware Templates เพื่อป้องกัน Language Mixing
    """
    # ตรวจจับภาษาจาก Template ก่อน
    lang = detect_language_from_template(template)
    
    # Reversal templates per language
    reversal_map = {
        "en": {
            "founder_of": "{tail} was founded by {head}",
            "ceo_of": "The CEO of {tail} is {head}",
            "developed": "{tail} was developed by {head}"
        },
        "zh": {
            "founder_of": "{tail}是由{head}創立的",
            "ceo_of": "{tail}的執行長是{head}",
            "developed": "{tail}是由{head}開發的"
        },
        "ja": {
            "founder_of": "{tail}は{head}によって設立された",
            "ceo_of": "{tail}のCEOは{head}だ",
            "developed": "{tail}は{head}によって開発された"
        },
        "ko": {
            "founder_of": "{tail}은 {head}에 의해 설립되었다",
            "ceo_of": "{tail}의 CEO는 {head}이다",
            "developed": "{tail}은 {head}에 의해 개발되었다"
        },
        "th": {
            "founder_of": "{tail} ก่อตั้งโดย {head}",
            "ceo_of": "ซีอีโอของ {tail} คือ {head}",
            "developed": "{tail} พัฒนาโดย {head}"
        }
    }
    
    # ใช้ English เป็น fallback
    lang_map = reversal_map.get(lang, reversal_map["en"])
    
    new_template = template
    new_rels = relation_defs
    
    if random.random() < 0.3: # 30% สลับโครงสร้าง
        for rel_type, new_fmt in lang_map.items():
            for i, (h, t, r) in enumerate(relation_defs):
                if r == rel_type:
                    # สร้าง Template ใหม่โดยอ้างอิง Placeholder เดิม
                    new_template = new_fmt.format(head="{" + h + "}", tail="{" + t + "}")
                    break
    
    return new_template, entity_defs, new_rels


def get_label_description(label: str, label_type: str = "entity") -> str:
    """Retrieves a description for a given label."""
    description = ""
    
    if label_type == "entity":
        # 1. Try formal definition from ENTITY_TYPES
        if label in ENTITY_TYPES:
            description = ENTITY_TYPES[label]
        # 2. Try synonyms list if no formal definition
        elif label in ENTITY_LABEL_SYNONYMS:
             description = f"An entity referring to {', '.join(ENTITY_LABEL_SYNONYMS[label])}."
        # 3. If label itself is a phrase string (from synonym augmentation)
        # We assume the caller passes the CANONICAL label. 
            
    elif label_type == "relation":
        # 1. Try formal definition from RELATION_TYPES (value is tuple: (head, tail, desc))
        if label in RELATION_TYPES:
            description = RELATION_TYPES[label][2]
        # 2. Try synonyms list
        elif label in RELATION_LABEL_SYNONYMS:
            description = f"A relation indicating {', '.join(RELATION_LABEL_SYNONYMS[label])}."
            
    return description


def get_interleaving_noise(entities_dict: Dict, lang: str = "en") -> str:
    """สร้าง Entity ขวาง (Interleaving) ที่ไม่เกี่ยวข้องกัน
    
    🔥 FIX: Language-Aware Noise เพื่อป้องกัน Language Mixing
    """
    fake_type = random.choice(["person", "company", "city"])
    fake_val = get_entity(entities_dict, fake_type)
    
    distractors_by_lang = {
        "en": [
            f", along with {fake_val},",
            f" (while {fake_val} was absent)",
            f" and {fake_val}"
        ],
        "zh": [
            f"，與{fake_val}一起，",
            f"（{fake_val}不在場時）",
            f"和{fake_val}"
        ],
        "ja": [
            f"、{fake_val}と共に、",
            f"（{fake_val}が不在の間）",
            f"と{fake_val}"
        ],
        "ko": [
            f", {fake_val}와 함께,",
            f" ({fake_val} 부재 중)",
            f"과 {fake_val}"
        ],
        "th": [
            f" พร้อมกับ {fake_val}",
            f" (ขณะที่ {fake_val} ไม่อยู่)",
            f" และ {fake_val}"
        ]
    }
    
    distractors = distractors_by_lang.get(lang, distractors_by_lang["en"])
    return random.choice(distractors)


# -----------------------------------------------------------------
#  generate_sample function
# -----------------------------------------------------------------

def detect_language_from_template(template: str) -> str:
    """ตรวจจับภาษาจาก Template"""
    if any('\u4e00' <= char <= '\u9fff' for char in template):
        return 'zh'
    elif any('\u3040' <= char <= '\u30ff' for char in template):
        return 'ja'
    elif any('\uac00' <= char <= '\ud7af' for char in template):
        return 'ko'
    elif any('\u0e00' <= char <= '\u0e7f' for char in template):
        return 'th'
    return 'en'

def generate_sample(templates: List, entities_dict: Dict, use_style_variation: bool = True) -> Dict:
    template_raw, entity_defs, relation_defs = random.choice(templates)
    
    # ตรวจจับภาษา
    lang = detect_language_from_template(template_raw)
    
    # [NEW] Style Variation: 30% โอกาสใช้ Template แบบ Style ต่างๆ
    applied_style = None
    if use_style_variation and random.random() < 0.3 and relation_defs:
        # ลองหา Styled Template สำหรับ Relation แรก
        first_rel = relation_defs[0][2] if len(relation_defs) > 0 else None
        if first_rel:
            styled_template, applied_style = get_styled_template(first_rel, lang)
            if styled_template:
                template_raw = styled_template

    # [แทรกจุดที่ 1] สลับโครงสร้าง Active/Passive ก่อนเริ่มกระบวนการอื่น
    template_raw, entity_defs, relation_defs = apply_semantic_reversal(template_raw, entity_defs, relation_defs)
    
    # 1. Synonym Replacement ในตัว Template เอง
    template = apply_synonym(template_raw)
    
    replacements = {}
    used_values = set()
    
    # [NEW] Label Mapping Augmentation: เก็บ Mapping สำหรับ Cross-Label
    label_mappings = {}
    
    # 2. เตรียมข้อมูลสำหรับแต่ละ Placeholder
    for placeholder, entity_type in entity_defs:
        key = "{" + placeholder + "}"
        original_value = get_entity(entities_dict, entity_type)
        
        # ป้องกันค่าซ้ำ
        while original_value in used_values:
            original_value = get_entity(entities_dict, entity_type)
        
        # 🔥 ZERO-SHOT: Cross-Label Mapping: 40% โอกาสเปลี่ยน Label (เพิ่มจาก 20%)
        augmented_label = apply_label_mapping_augmentation(entity_type, probability=0.40)
        label_mappings[placeholder] = {
            "original": entity_type,
            "augmented": augmented_label
        }
        
        # --- APPLY NOISE STRATEGIES ---
        processed_value = original_value

        # 1. เช็คว่าเป็นภาษาจีนหรือไม่
        # เช็คว่าเป็นภาษาไทยหรือไม่
        is_thai = any('\u0e00' <= char <= '\u0e7f' for char in processed_value)
        is_chinese = any('\u4e00' <= char <= '\u9fff' for char in processed_value)

        if is_thai:
            processed_value = apply_thai_slang_shorten(processed_value, probability=0.2)
            processed_value = apply_thai_vowel_noise(processed_value, probability=0.15)
            processed_value = apply_thai_keyboard_shift(processed_value, probability=0.05)
        elif is_chinese:
            processed_value = apply_chinese_incomplete(processed_value, probability=0.2)
            processed_value = apply_chinese_noise(processed_value, probability=0.15)
        else:
            # ภาษาอังกฤษ
            processed_value = apply_incomplete_entity(processed_value, probability=0.1)
            processed_value = apply_typo(processed_value, probability=0.1)
        
        # Apply Incomplete (เฉพาะบางครั้ง)
        processed_value = apply_incomplete_entity(processed_value, probability=0.15)
        
        # Apply Typo (เฉพาะภาษาอังกฤษ/โรมัน)
        if any(c.isalpha() for c in processed_value):
            processed_value = apply_typo(processed_value, probability=0.1)
            processed_value = apply_case_variation(processed_value, probability=0.2)
        
        replacements[key] = processed_value
        used_values.add(original_value) # เก็บต้นฉบับไว้เช็คซ้ำ


    # [แทรกจุดที่ 2] แทรกชื่อขวาง (Interleaving) ก่อนจะรวมร่างเป็นประโยค
    if random.random() < 0.2: # ใส่โอกาส 20%
        noise = get_interleaving_noise(entities_dict, lang)  # 🔥 FIX: Pass language
        first_ph_key = "{" + entity_defs[0][0] + "}"
        template = template.replace(first_ph_key, first_ph_key + noise)
        
    # 3. Build text
    text = template
    for key, value in replacements.items():
        text = text.replace(key, value)


    # [แทรกจุดที่ 3] เปลี่ยนชื่อที่ซ้ำให้เป็นสรรพนาม (He/She/It)
    for placeholder, entity_type in entity_defs:
        val = replacements["{" + placeholder + "}"]
        text = apply_indirect_reference(text, val, entity_type)

    # 4. [NEW] AMBIGUOUS TERMS / LOOK-ALIKES (Hard Negative Mining)
    # เพิ่ม "ชื่อหลอก" ที่หน้าตาเหมือน Entity เข้าไปในประโยคแต่ไม่ Mark label
    # 🔥 FIX: Language-Aware Filler Phrases เพื่อป้องกัน Language Mixing
    if random.random() < 0.3: # 30% ของข้อมูลจะมีตัวหลอก
        random_type = random.choice(list(entities_dict.keys()))
        fake_entity = get_entity(entities_dict, random_type)
        if fake_entity not in used_values:
            filler_phrases_by_lang = {
                "en": [
                    f" (similar to {fake_entity})",
                    f" unlike {fake_entity}",
                    f". Note: {fake_entity} was not involved."
                ],
                "zh": [
                    f"（與{fake_entity}相似）",
                    f"，與{fake_entity}不同",
                    f"。注意：{fake_entity}並未參與。"
                ],
                "ja": [
                    f"（{fake_entity}に似ている）",
                    f"、{fake_entity}とは異なり",
                    f"。注：{fake_entity}は関与していない。"
                ],
                "ko": [
                    f" ({fake_entity}와 유사)",
                    f", {fake_entity}와 달리",
                    f". 참고: {fake_entity}은 관련되지 않았다."
                ],
                "th": [
                    f" (คล้ายกับ {fake_entity})",
                    f" ต่างจาก {fake_entity}",
                    f" หมายเหตุ: {fake_entity} ไม่ได้เกี่ยวข้อง"
                ]
            }
            filler_phrases = filler_phrases_by_lang.get(lang, filler_phrases_by_lang["en"])
            text += random.choice(filler_phrases)


    # [แทรกจุดที่ 4] ตัดสินใจว่าจะเป็นเคส "ไม่มีความสัมพันธ์" หรือไม่
    # 🔥 FIX: เพิ่มเป็น 50% เพื่อลดการเดามั่ว (เพิ่ม Precision ของ RE)
    is_negative_case = random.random() < 0.50

    # 5. Build entities metadata
    entities = []
    for placeholder, entity_type in entity_defs:
        key = "{" + placeholder + "}"
        entity_text = replacements[key]
        
        # [NEW] ใช้ Augmented Label ถ้ามี (Cross-Label Mapping)
        final_label = entity_type
        if placeholder in label_mappings:
            # 50% โอกาสใช้ Augmented Label, 50% ใช้ Original
            if random.random() < 0.5:
                final_label = label_mappings[placeholder]["augmented"]
        
        # 🔥 Capture Canonical Label for Description (ก่อนจะถูกเปลี่ยนเป็น Synonym)
        canonical_label_for_desc = final_label

        # 🔥 V3: Label Synonym Augmentation - 50% โอกาสใช้ synonym
        final_label = get_label_synonym(final_label, label_type="entity", probability=0.5)

        # รับ Description (ใช้ canonical label ในการหา)
        description = get_label_description(canonical_label_for_desc, label_type="entity")
        
        # ค้นหาตำแหน่งที่ถูกต้อง (ระวังคำซ้ำ)
        start = text.find(entity_text)
        if start != -1:
            entities.append({
                "start": start,
                "end": start + len(entity_text),
                "label": final_label,
                "text": entity_text,
                "description": description
            })

    # Build relations
    relations = []
    # [แก้ไขจุดที่ 5] ตรวจสอบว่าทั้ง Head และ Tail มีตัวตนอยู่ใน Text จริงๆ
    if not is_negative_case:
        for head_ph, tail_ph, rel_type in relation_defs:
            head_key, tail_key = "{" + head_ph + "}", "{" + tail_ph + "}"
            
            if head_key in replacements and tail_key in replacements:
                head_text = replacements[head_key]
                tail_text = replacements[tail_key]
                
                # 🔥 หัวใจสำคัญ: ต้องเจอทั้งคู่ในประโยคที่ถูก Noise แล้วเท่านั้น
                if head_text in text and tail_text in text:
                    # 🔥 ZERO-SHOT: Canonicalize Label (Consolidate Labels)
                    final_rel_type = canonicalize_relation_label(rel_type)
                    
                    # 🔥 Capture Canonical Label for Description
                    canonical_rel_for_desc = final_rel_type

                    # 🔥 V3: Label Synonym Augmentation for Relations - 50% โอกาสใช้ synonym
                    final_rel_type = get_label_synonym(final_rel_type, label_type="relation", probability=0.5)
                    
                    # รับ Description
                    rel_description = get_label_description(canonical_rel_for_desc, label_type="relation")

                    relations.append({
                        "head": head_text,
                        "tail": tail_text,
                        "label": final_rel_type,
                        "description": rel_description
                    })

    # ============================================================
    # 🎯 วางตำแหน่งใหม่ตรงนี้ (ก่อน Padding)
    # ============================================================
    
    sample_data = {"text": text, "entities": entities, "relations": relations}
    
    # 1. เพิ่มความซับซ้อนของสรรพนาม (Coreference) - ฟังก์ชันนี้จะเพิ่ม text และ metadata
    text, entities, relations = apply_coreference_logic(sample_data)
    
    # 2. เพิ่มตัวหลอก (Hard Negatives) - ฟังก์ชันนี้จะแทรก noise และขยับ Index เก่าให้อัตโนมัติ
    text, entities = add_hard_negatives({"text": text, "entities": entities}, entities_dict)

    # ============================================================


    text, offset = apply_contextual_padding(text)
    
    # Adjust entity positions based on offset
    for ent in entities:
        ent["start"] += offset
        ent["end"] += offset
    
    return {"text": text, "entities": entities, "relations": relations}


def generate_samples(templates: List, entities_dict: Dict, count: int, use_style_variation: bool = True) -> List[Dict]:
    """Generate multiple samples with optional style variation."""
    samples = []
    for _ in range(count):
        sample = generate_sample(templates, entities_dict, use_style_variation=use_style_variation)
        if sample["entities"] and sample["relations"]:  # Only add valid samples
            samples.append(sample)
    return samples


def generate_dataset(target_count: int = 10000) -> List[Dict]:
    """Generate a balanced multilingual dataset with Zero-Shot Generalization focus."""
    
    # Distribution: 30% EN, 25% ZH, 20% JA, 15% KO, 10% TH
    en_count = int(target_count * 0.30)
    zh_count = int(target_count * 0.25)
    ja_count = int(target_count * 0.20)
    ko_count = int(target_count * 0.15)
    th_count = int(target_count * 0.10)
    
    print(f"Generating {en_count} English samples...")
    en_samples = generate_samples(EN_TEMPLATES, EN_ENTITIES, en_count)
    
    print(f"Generating {zh_count} Chinese samples...")
    zh_samples = generate_samples(ZH_TEMPLATES, ZH_ENTITIES, zh_count)
    
    print(f"Generating {ja_count} Japanese samples...")
    ja_samples = generate_samples(JA_TEMPLATES, JA_ENTITIES, ja_count)
    
    print(f"Generating {ko_count} Korean samples...")
    ko_samples = generate_samples(KO_TEMPLATES, KO_ENTITIES, ko_count)
    
    print(f"Generating {th_count} Thai samples...")
    th_samples = generate_samples(TH_TEMPLATES, TH_ENTITIES, th_count)

    print(f"Generating {en_count//10} English complex samples...")
    en_samples += generate_samples(EN_COMPLEX_TEMPLATES, EN_ENTITIES, en_count // 10)

    print(f"Generating {zh_count//10} Chinese complex samples...")
    zh_samples += generate_samples(ZH_COMPLEX_TEMPLATES, ZH_ENTITIES, zh_count // 10)

    print(f"Generating {th_count//10} Thai complex samples...")
    th_samples += generate_samples(TH_COMPLEX_TEMPLATES, TH_ENTITIES, th_count // 10)
    
    # 🔥 ZERO-SHOT: Add Generic Label Templates (15% of EN count)
    generic_count = int(en_count * 0.15)
    print(f"🔥 Generating {generic_count} Generic Label samples for Zero-Shot...")
    generic_samples = generate_samples(GENERIC_ENTITY_TEMPLATES, EN_ENTITIES, generic_count)
    
    # 🔥 ZERO-SHOT: Add CoNLL04-style Templates (10% of EN count)  
    conll_count = int(en_count * 0.10)
    print(f"🔥 Generating {conll_count} CoNLL04-style samples for Zero-Shot...")
    conll_samples = generate_samples(CONLL04_STYLE_TEMPLATES, EN_ENTITIES, conll_count)
    
    # 🔥 LINGUISTIC PARAPHRASING: Passive Voice, Appositive, Relative Clause (20% of EN count)
    paraphrase_count = int(en_count * 0.20)
    print(f"🔥 Generating {paraphrase_count} Linguistic Paraphrase samples (Passive/Appositive/Relative)...")
    paraphrase_samples = generate_samples(LINGUISTIC_PARAPHRASE_TEMPLATES, EN_ENTITIES, paraphrase_count)
    
    # 🔥 V3: CROSS-RE STYLE Templates for Zero-Shot Generalization (15% of EN count)
    cross_re_count = int(en_count * 0.15)
    print(f"🔥 Generating {cross_re_count} Cross-RE-style samples (politicalparty, band, musicalartist, etc.)...")
    cross_re_samples = generate_samples(CROSS_RE_STYLE_TEMPLATES, EN_ENTITIES, cross_re_count)
    
    # Combine and shuffle
    all_samples = en_samples + zh_samples + ja_samples + ko_samples + th_samples + generic_samples + conll_samples + paraphrase_samples + cross_re_samples
    random.shuffle(all_samples)
    
    return all_samples


def print_statistics(samples: List[Dict]):
    """Print dataset statistics."""
    total_entities = 0
    total_relations = 0
    entity_types = {}
    relation_types = {}
    
    for sample in samples:
        total_entities += len(sample["entities"])
        total_relations += len(sample["relations"])
        
        for entity in sample["entities"]:
            label = entity["label"]
            entity_types[label] = entity_types.get(label, 0) + 1
        
        for relation in sample["relations"]:
            label = relation["label"]
            relation_types[label] = relation_types.get(label, 0) + 1
    
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples: {len(samples)}")
    print(f"Total entities: {total_entities}")
    print(f"Total relations: {total_relations}")
    print(f"Avg entities/sample: {total_entities / len(samples):.2f}")
    print(f"Avg relations/sample: {total_relations / len(samples):.2f}")
    
    print(f"\nEntity Types ({len(entity_types)} unique):")
    for label, count in sorted(entity_types.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count} ({count/total_entities*100:.1f}%)")
    
    print(f"\nRelation Types ({len(relation_types)} unique):")
    for label, count in sorted(relation_types.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count} ({count/total_relations*100:.1f}%)")


def print_type_summary():
    """Print summary of all entity and relation types."""
    print("\n" + "=" * 60)
    print(f"ENTITY TYPES: {len(ENTITY_TYPES)} types")
    print("=" * 60)
    for i, (etype, desc) in enumerate(ENTITY_TYPES.items(), 1):
        print(f"  {i:3d}. {etype}: {desc}")
    
    print("\n" + "=" * 60)
    print(f"RELATION TYPES: {len(RELATION_TYPES)} types")
    print("=" * 60)
    for i, (rtype, (head, tail, desc)) in enumerate(RELATION_TYPES.items(), 1):
        print(f"  {i:3d}. {rtype}: {head} -> {tail} ({desc})")
    
    print("\n" + "=" * 60)
    print("CROSS-LABEL MAPPING (Hierarchical)")
    print("=" * 60)
    for parent, children in LABEL_HIERARCHY.items():
        print(f"  {parent} → {children}")
    
    print("\n" + "=" * 60)
    print("🔥 RELATION ALIASES (Zero-Shot)")
    print("=" * 60)
    for canonical, aliases in RELATION_ALIASES.items():
        print(f"  {canonical} = {aliases}")
    
    print("\n" + "=" * 60)
    print("🔥 ZERO-SHOT TEMPLATES")
    print("=" * 60)
    print(f"  GENERIC_ENTITY_TEMPLATES: {len(GENERIC_ENTITY_TEMPLATES)} templates")
    print(f"  CONLL04_STYLE_TEMPLATES: {len(CONLL04_STYLE_TEMPLATES)} templates")
    print(f"  LINGUISTIC_PARAPHRASE_TEMPLATES: {len(LINGUISTIC_PARAPHRASE_TEMPLATES)} templates (Passive/Appositive/Relative)")
    
    print("\n" + "=" * 60)
    print("SENTENCE STYLES")
    print("=" * 60)
    for style in SENTENCE_STYLES.keys():
        print(f"  - {style}")


if __name__ == "__main__":
    
    # if True:
    #     print_type_summary()
    #     exit(0)
    
    random.seed(42)
    
    print(f"NERRE Dataset Generator v2 - 🔥 ZERO-SHOT READY")
    print(f"Entity Types: {len(ENTITY_TYPES)}")
    print(f"Relation Types: {len(RELATION_TYPES)}")
    print(f"Cross-Label Mappings: {len(LABEL_HIERARCHY)}")
    print(f"Relation Aliases: {len(RELATION_ALIASES)}")
    print(f"Generic Templates: {len(GENERIC_ENTITY_TEMPLATES)} + {len(CONLL04_STYLE_TEMPLATES)}")
    print(f"Linguistic Paraphrase: {len(LINGUISTIC_PARAPHRASE_TEMPLATES)} (Passive/Appositive/Relative)")
    print(f"Sentence Styles: {list(SENTENCE_STYLES.keys())}")
    print(f"Generating Train/Val/Test Datasets...")

    # Configuration
    SPLITS = {
        "train": 80000,
        "val": 10000,
        "test": 10000
    }
    BASE_PATH = "/data/tcustpg18/NERRE/NERRE/dataset"
    VERSION = "v8"

    for split_name, count in SPLITS.items():
        print("\n" + "=" * 60)
        print(f"Generating {split_name.upper()} Set ({count} samples)...")
        print("=" * 60)
        
        samples = generate_dataset(count)
        print_statistics(samples)
        
        output_path = f"{BASE_PATH}/multilingual_data_{VERSION}_{split_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved {split_name} dataset to: {output_path}")
