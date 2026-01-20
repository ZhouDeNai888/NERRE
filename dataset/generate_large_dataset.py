#!/usr/bin/env python3
"""
Generate a large multilingual training dataset for NERRE.
Target: 5000+ samples across 5 languages (English, Chinese, Japanese, Korean, Thai)
"""

import json
import random
from typing import List, Dict, Any

# ============================================================================
# Entity Database - Real-world entities for each language
# ============================================================================

# English entities
EN_PERSONS = [
    "Elon Musk", "Bill Gates", "Steve Jobs", "Jeff Bezos", "Mark Zuckerberg",
    "Tim Cook", "Satya Nadella", "Sundar Pichai", "Larry Page", "Sergey Brin",
    "Warren Buffett", "Sam Altman", "Jensen Huang", "Lisa Su", "Pat Gelsinger",
    "Andy Jassy", "Brian Chesky", "Jack Dorsey", "Reid Hoffman", "Peter Thiel",
    "Paul Allen", "Steve Wozniak", "Larry Ellison", "Michael Dell", "Marc Benioff",
    "Travis Kalanick", "Dara Khosrowshahi", "Daniel Ek", "Drew Houston", "Kevin Systrom",
    "Jan Koum", "Brian Acton", "Evan Spiegel", "Bobby Murphy", "Stewart Butterfield",
    "Eric Yuan", "Tony Hsieh", "Reed Hastings", "Marc Randolph", "Brian Krzanich",
    "Ginni Rometty", "Meg Whitman", "Carly Fiorina", "Marissa Mayer", "Sheryl Sandberg",
    "Susan Wojcicki", "Ruth Porat", "Safra Catz", "Arvind Krishna", "Jim Whitehurst",
    "Linus Torvalds", "Guido van Rossum", "Brendan Eich", "James Gosling", "Dennis Ritchie",
    "Ken Thompson", "Bjarne Stroustrup", "Anders Hejlsberg", "John Carmack", "Gabe Newell"
]

EN_ORGS = [
    "Apple", "Microsoft", "Google", "Amazon", "Meta", "Tesla", "SpaceX", "NVIDIA",
    "Intel", "AMD", "IBM", "Oracle", "Salesforce", "Adobe", "Netflix", "Spotify",
    "Uber", "Airbnb", "Twitter", "LinkedIn", "PayPal", "Square", "Stripe", "Shopify",
    "Zoom", "Slack", "Dropbox", "Instagram", "WhatsApp", "Snapchat", "Pinterest",
    "TikTok", "Reddit", "Discord", "Twitch", "GitHub", "GitLab", "Atlassian", "Databricks",
    "Snowflake", "Palantir", "Coinbase", "Robinhood", "OpenAI", "Anthropic", "DeepMind",
    "Boston Dynamics", "Blue Origin", "Virgin Galactic", "Rivian", "Lucid Motors",
    "Ford", "General Motors", "Toyota", "Honda", "Volkswagen", "BMW", "Mercedes-Benz",
    "Samsung", "Sony", "Panasonic", "LG", "Huawei", "Xiaomi", "Lenovo", "Dell", "HP"
]

EN_LOCATIONS = [
    "California", "San Francisco", "Silicon Valley", "Palo Alto", "Mountain View",
    "Cupertino", "Seattle", "New York", "Boston", "Austin", "Denver", "Chicago",
    "Los Angeles", "San Diego", "Portland", "Phoenix", "Miami", "Atlanta", "Dallas",
    "Houston", "Washington DC", "London", "Berlin", "Paris", "Tokyo", "Singapore",
    "Hong Kong", "Shanghai", "Beijing", "Seoul", "Sydney", "Melbourne", "Toronto",
    "Vancouver", "Tel Aviv", "Bangalore", "Mumbai", "Amsterdam", "Dublin", "Zurich",
    "Stockholm", "Helsinki", "Copenhagen", "Oslo", "Vienna", "Prague", "Warsaw",
    "Barcelona", "Madrid", "Rome", "Milan", "São Paulo", "Mexico City", "Buenos Aires"
]

EN_PRODUCTS = [
    "iPhone", "iPad", "MacBook", "Apple Watch", "AirPods", "Windows", "Office", "Xbox",
    "Azure", "Teams", "Chrome", "Android", "Gmail", "YouTube", "AWS", "Alexa", "Kindle",
    "Fire TV", "Facebook", "Instagram", "WhatsApp", "Messenger", "Model S", "Model 3",
    "Cybertruck", "Starlink", "GeForce", "CUDA", "Photoshop", "Premiere", "Illustrator",
    "LinkedIn", "GitHub", "Bing", "GPT-4", "ChatGPT", "Claude", "Gemini", "LLaMA",
    "TensorFlow", "PyTorch", "React", "Angular", "Vue", "Node.js", "Docker", "Kubernetes",
    "Spotify", "Netflix", "Uber", "Airbnb", "Slack", "Zoom", "Dropbox", "Notion",
    "PlayStation", "Nintendo Switch", "Steam", "Unity", "Unreal Engine", "Java", "Python"
]

EN_PROGRAMLANG = [
    "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust",
    "Swift", "Kotlin", "Ruby", "PHP", "Perl", "R", "MATLAB", "Julia", "Scala",
    "Haskell", "Clojure", "Elixir", "Erlang", "F#", "Dart", "Lua", "Objective-C"
]

# Chinese (Traditional + Simplified) entities
ZH_PERSONS = [
    "張忠謀", "郭台銘", "施振榮", "施崇棠", "林百里", "王雪紅", "蔡明介", "劉德音",
    "馬雲", "馬化騰", "李彥宏", "任正非", "雷軍", "劉強東", "張一鳴", "黃錚",
    "王興", "程維", "楊元慶", "柳傳志", "董明珠", "張瑞敏", "曹德旺", "宗慶後",
    "馬克·祖克柏", "伊隆·馬斯克", "比爾蓋茲", "史蒂夫·賈伯斯", "傑夫·貝佐斯",
    "保羅艾倫", "蒂姆·庫克", "孫正義", "稲盛和夫", "柳井正", "三木谷浩史"
]

ZH_ORGS = [
    "台積電", "鴻海", "聯發科", "華碩", "宏碁", "廣達", "仁寶", "和碩", "緯創",
    "聯電", "日月光", "台達電", "國泰金", "富邦金", "中信金", "統一企業", "長榮",
    "阿里巴巴", "騰訊", "百度", "華為", "小米", "京東", "美團", "字節跳動",
    "拼多多", "滴滴", "網易", "攜程", "聯想", "格力", "海爾", "比亞迪", "寧德時代",
    "中國移動", "中國電信", "中國聯通", "工商銀行", "建設銀行", "農業銀行",
    "蘋果公司", "微軟", "谷歌", "亞馬遜", "特斯拉", "英偉達", "英特爾", "三星電子"
]

ZH_LOCATIONS = [
    "台北", "新竹", "台中", "台南", "高雄", "桃園", "新北", "基隆", "花蓮",
    "北京", "上海", "深圳", "杭州", "廣州", "成都", "武漢", "南京", "蘇州",
    "天津", "重慶", "西安", "青島", "大連", "廈門", "長沙", "鄭州", "東莞",
    "香港", "澳門", "矽谷", "加州", "紐約", "西雅圖", "東京", "首爾", "新加坡"
]

ZH_PRODUCTS = [
    "微信", "QQ", "抖音", "TikTok", "支付寶", "淘寶", "天貓", "京東商城",
    "美團外賣", "滴滴出行", "百度搜索", "小紅書", "B站", "微博", "知乎",
    "iPhone", "MacBook", "iPad", "Windows", "Office", "Azure", "Android",
    "華為Mate", "小米手機", "OPPO", "vivo", "榮耀", "紅米", "比亞迪電動車"
]

# Japanese entities
JA_PERSONS = [
    "盛田昭夫", "井深大", "本田宗一郎", "豊田喜一郎", "松下幸之助",
    "山内房治郎", "宮本茂", "岩田聡", "孫正義", "三木谷浩史",
    "柳井正", "稲盛和夫", "安藤百福", "鳥山明", "宮崎駿",
    "任天堂", "堀井雄二", "坂口博信", "小島秀夫", "名越稔洋"
]

JA_ORGS = [
    "ソニー", "トヨタ", "ホンダ", "任天堂", "パナソニック", "日立",
    "東芝", "シャープ", "キヤノン", "ニコン", "富士通", "NEC",
    "ソフトバンク", "楽天", "ユニクロ", "日産", "スズキ", "マツダ",
    "ヤマハ", "セガ", "バンダイナムコ", "カプコン", "スクウェア・エニックス",
    "コナミ", "フロムソフトウェア", "サイバーエージェント", "DeNA", "LINE",
    "メルカリ", "スマートニュース", "Preferred Networks", "三菱", "住友"
]

JA_LOCATIONS = [
    "東京", "大阪", "京都", "名古屋", "福岡", "横浜", "神戸", "札幌",
    "広島", "仙台", "川崎", "浜松", "愛知県", "静岡県", "千葉", "埼玉",
    "北海道", "沖縄", "奈良", "鎌倉", "秋葉原", "渋谷", "新宿", "六本木"
]

JA_PRODUCTS = [
    "PlayStation", "Nintendo Switch", "ウォークマン", "プリウス",
    "ファミコン", "スーパーファミコン", "ゲームボーイ", "Wii",
    "ゼルダの伝説", "マリオ", "ポケモン", "ファイナルファンタジー",
    "ドラゴンクエスト", "メタルギア", "ダークソウル", "VAIO",
    "カップヌードル", "LINE", "メルカリ", "PayPay", "楽天市場"
]

# Korean entities
KO_PERSONS = [
    "이병철", "이건희", "이재용", "정주영", "정몽구", "정의선",
    "최태원", "구본무", "구광모", "신동빈", "김범수", "이해진",
    "방시혁", "양현석", "이수만", "박진영", "봉준호", "손석희"
]

KO_ORGS = [
    "삼성전자", "현대자동차", "SK하이닉스", "LG전자", "포스코",
    "한화", "롯데", "신세계", "카카오", "네이버", "쿠팡", "배달의민족",
    "하이브", "YG엔터테인먼트", "SM엔터테인먼트", "JYP엔터테인먼트",
    "CJ", "KT", "SK텔레콤", "LG화학", "현대중공업", "대우", "기아"
]

KO_LOCATIONS = [
    "서울", "부산", "인천", "대구", "광주", "대전", "울산", "수원",
    "성남", "분당", "판교", "제주", "경기도", "강남", "홍대", "이태원"
]

KO_PRODUCTS = [
    "갤럭시", "카카오톡", "네이버", "쿠팡", "배달의민족", "토스",
    "멜론", "지마켓", "11번가", "라인", "현대차", "기아차",
    "삼성페이", "카카오페이", "네이버페이", "당근마켓", "무신사"
]

# Thai entities
TH_PERSONS = [
    "ธนินท์ เจียรวนนท์", "เจริญ สิริวัฒนภักดี", "สมคิด จาตุศรีพิทักษ์",
    "ชิน โสภณพนิช", "วิชัย ศรีวัฒนประภา", "ธนาธร จึงรุ่งเรืองกิจ"
]

TH_ORGS = [
    "เครือเจริญโภคภัณฑ์", "ปตท.", "ธนาคารกรุงเทพ", "เซ็นทรัล",
    "ไทยเบฟเวอเรจ", "บุญรอด", "ทรู", "AIS", "DTAC", "SCB",
    "กสิกรไทย", "ไทยพาณิชย์", "ปูนซิเมนต์ไทย", "เมเจอร์",
    "ซีพี ออลล์", "โลตัส", "บิ๊กซี", "เซ็นทรัลพัฒนา"
]

TH_LOCATIONS = [
    "กรุงเทพฯ", "เชียงใหม่", "ภูเก็ต", "พัทยา", "หาดใหญ่",
    "ขอนแก่น", "นครราชสีมา", "อุดรธานี", "ระยอง", "ชลบุรี",
    "ประเทศไทย", "สยาม", "สุขุมวิท", "สีลม", "รัชดา"
]

TH_PRODUCTS = [
    "ทรูมูฟ", "AIS", "DTAC", "LINE", "แกร็บ", "ลาซาด้า",
    "ช้อปปี้", "โรบินฮู้ด", "ไลน์แมน", "ฟู้ดแพนด้า"
]

# Date patterns
YEARS = list(range(1880, 2025))

# ============================================================================
# Template generators for each language
# ============================================================================

def generate_en_samples(count: int) -> List[Dict]:
    """Generate English samples."""
    templates = [
        # founder_of + founded_in + located_in
        ("{person} founded {org} in {year} in {location}.",
         [("person", "{person}"), ("organisation", "{org}"), ("date", "{year}"), ("location", "{location}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{year}", "founded_in"), ("{org}", "{location}", "located_in")]),
        
        # ceo_of
        ("{person} is the CEO of {org}.",
         [("person", "{person}"), ("organisation", "{org}")],
         [("{person}", "{org}", "ceo_of")]),
        
        ("{person} serves as CEO of {org}, which is headquartered in {location}.",
         [("person", "{person}"), ("organisation", "{org}"), ("location", "{location}")],
         [("{person}", "{org}", "ceo_of"), ("{org}", "{location}", "located_in")]),
        
        # developed
        ("{org} developed {product} in {year}.",
         [("organisation", "{org}"), ("product", "{product}"), ("date", "{year}")],
         [("{org}", "{product}", "developed"), ("{product}", "{year}", "released_in")]),
        
        ("{product} was created by {org}.",
         [("product", "{product}"), ("organisation", "{org}")],
         [("{org}", "{product}", "developed")]),
        
        # creator_of (for programming languages)
        ("{person} created {programlang}.",
         [("person", "{person}"), ("programlang", "{programlang}")],
         [("{person}", "{programlang}", "creator_of")]),
        
        ("{person} is the creator of {programlang}, which was released in {year}.",
         [("person", "{person}"), ("programlang", "{programlang}"), ("date", "{year}")],
         [("{person}", "{programlang}", "creator_of"), ("{programlang}", "{year}", "released_in")]),
        
        # Complex sentences
        ("{person1} and {person2} co-founded {org} in {year}.",
         [("person", "{person1}"), ("person", "{person2}"), ("organisation", "{org}"), ("date", "{year}")],
         [("{person1}", "{org}", "founder_of"), ("{person2}", "{org}", "founder_of"), ("{org}", "{year}", "founded_in")]),
        
        ("{org} was founded by {person} in {location}. The company developed {product}.",
         [("organisation", "{org}"), ("person", "{person}"), ("location", "{location}"), ("product", "{product}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{location}", "located_in"), ("{org}", "{product}", "developed")]),
        
        ("The headquarters of {org} is located in {location}.",
         [("organisation", "{org}"), ("location", "{location}")],
         [("{org}", "{location}", "located_in")]),
        
        ("{person} became the CEO of {org} after founding {org2}.",
         [("person", "{person}"), ("organisation", "{org}"), ("organisation", "{org2}")],
         [("{person}", "{org}", "ceo_of"), ("{person}", "{org2}", "founder_of")]),
        
        ("{org} released {product} in {year}, which revolutionized the industry.",
         [("organisation", "{org}"), ("product", "{product}"), ("date", "{year}")],
         [("{org}", "{product}", "developed"), ("{product}", "{year}", "released_in")]),
        
        ("In {year}, {person} established {org} in {location}.",
         [("date", "{year}"), ("person", "{person}"), ("organisation", "{org}"), ("location", "{location}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{year}", "founded_in"), ("{org}", "{location}", "located_in")]),
        
        ("{product} is a product by {org}, released in {year}.",
         [("product", "{product}"), ("organisation", "{org}"), ("date", "{year}")],
         [("{org}", "{product}", "developed"), ("{product}", "{year}", "released_in")]),
    ]
    
    samples = []
    for _ in range(count):
        template, entity_templates, relation_templates = random.choice(templates)
        
        # Fill in placeholders
        replacements = {
            "{person}": random.choice(EN_PERSONS),
            "{person1}": random.choice(EN_PERSONS),
            "{person2}": random.choice(EN_PERSONS),
            "{org}": random.choice(EN_ORGS),
            "{org2}": random.choice(EN_ORGS),
            "{location}": random.choice(EN_LOCATIONS),
            "{product}": random.choice(EN_PRODUCTS),
            "{programlang}": random.choice(EN_PROGRAMLANG),
            "{year}": str(random.choice(YEARS)),
        }
        
        # Make sure person1 != person2, org != org2
        while replacements["{person1}"] == replacements["{person2}"]:
            replacements["{person2}"] = random.choice(EN_PERSONS)
        while replacements["{org}"] == replacements["{org2}"]:
            replacements["{org2}"] = random.choice(EN_ORGS)
        
        text = template
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, value)
        
        # Build entities
        entities = []
        for label, placeholder in entity_templates:
            entity_text = replacements[placeholder]
            start = text.find(entity_text)
            if start != -1:
                # Check if already added (avoid duplicates)
                if not any(e["text"] == entity_text and e["start"] == start for e in entities):
                    entities.append({
                        "start": start,
                        "end": start + len(entity_text),
                        "label": label,
                        "text": entity_text
                    })
        
        # Build relations
        relations = []
        for head_ph, tail_ph, rel_type in relation_templates:
            head_text = replacements[head_ph]
            tail_text = replacements[tail_ph]
            relations.append({
                "head": head_text,
                "tail": tail_text,
                "label": rel_type
            })
        
        samples.append({
            "text": text,
            "entities": entities,
            "relations": relations
        })
    
    return samples


def generate_zh_samples(count: int) -> List[Dict]:
    """Generate Chinese (Traditional + Simplified) samples."""
    templates = [
        # founder_of + founded_in + located_in
        ("{person}於{year}在{location}創立了{org}。",
         [("person", "{person}"), ("date", "{year}"), ("location", "{location}"), ("organisation", "{org}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{year}", "founded_in"), ("{org}", "{location}", "located_in")]),
        
        ("{org}由{person}於{year}創立。",
         [("organisation", "{org}"), ("person", "{person}"), ("date", "{year}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{year}", "founded_in")]),
        
        ("{org}由{person}創辦，總部設在{location}。",
         [("organisation", "{org}"), ("person", "{person}"), ("location", "{location}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{location}", "located_in")]),
        
        # ceo_of
        ("{person}是{org}的執行長。",
         [("person", "{person}"), ("organisation", "{org}")],
         [("{person}", "{org}", "ceo_of")]),
        
        ("{person}擔任{org}的CEO，公司位於{location}。",
         [("person", "{person}"), ("organisation", "{org}"), ("location", "{location}")],
         [("{person}", "{org}", "ceo_of"), ("{org}", "{location}", "located_in")]),
        
        # developed
        ("{org}開發了{product}。",
         [("organisation", "{org}"), ("product", "{product}")],
         [("{org}", "{product}", "developed")]),
        
        ("{product}由{org}於{year}推出。",
         [("product", "{product}"), ("organisation", "{org}"), ("date", "{year}")],
         [("{org}", "{product}", "developed"), ("{product}", "{year}", "released_in")]),
        
        # located_in
        ("{org}的總部位於{location}。",
         [("organisation", "{org}"), ("location", "{location}")],
         [("{org}", "{location}", "located_in")]),
        
        # Complex sentences
        ("{person}和{person2}於{year}共同創立了{org}。",
         [("person", "{person}"), ("person", "{person2}"), ("date", "{year}"), ("organisation", "{org}")],
         [("{person}", "{org}", "founder_of"), ("{person2}", "{org}", "founder_of"), ("{org}", "{year}", "founded_in")]),
        
        ("{org}成立於{year}，由{person}創立，總部在{location}。",
         [("organisation", "{org}"), ("date", "{year}"), ("person", "{person}"), ("location", "{location}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{year}", "founded_in"), ("{org}", "{location}", "located_in")]),
        
        ("{org}由{person}於{year}在{location}創立，開發了{product}。",
         [("organisation", "{org}"), ("person", "{person}"), ("date", "{year}"), ("location", "{location}"), ("product", "{product}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{year}", "founded_in"), ("{org}", "{location}", "located_in"), ("{org}", "{product}", "developed")]),
        
        ("{person}是{org}的創辦人兼執行長。",
         [("person", "{person}"), ("organisation", "{org}")],
         [("{person}", "{org}", "founder_of"), ("{person}", "{org}", "ceo_of")]),
        
        ("{year}，{person}在{location}創立了{org}。",
         [("date", "{year}"), ("person", "{person}"), ("location", "{location}"), ("organisation", "{org}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{year}", "founded_in"), ("{org}", "{location}", "located_in")]),
    ]
    
    samples = []
    for _ in range(count):
        template, entity_templates, relation_templates = random.choice(templates)
        
        replacements = {
            "{person}": random.choice(ZH_PERSONS),
            "{person2}": random.choice(ZH_PERSONS),
            "{org}": random.choice(ZH_ORGS),
            "{location}": random.choice(ZH_LOCATIONS),
            "{product}": random.choice(ZH_PRODUCTS),
            "{year}": str(random.choice(YEARS)) + "年",
        }
        
        while replacements["{person}"] == replacements["{person2}"]:
            replacements["{person2}"] = random.choice(ZH_PERSONS)
        
        text = template
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, value)
        
        entities = []
        for label, placeholder in entity_templates:
            entity_text = replacements[placeholder]
            start = text.find(entity_text)
            if start != -1:
                if not any(e["text"] == entity_text and e["start"] == start for e in entities):
                    entities.append({
                        "start": start,
                        "end": start + len(entity_text),
                        "label": label,
                        "text": entity_text
                    })
        
        relations = []
        for head_ph, tail_ph, rel_type in relation_templates:
            head_text = replacements[head_ph]
            tail_text = replacements[tail_ph]
            relations.append({
                "head": head_text,
                "tail": tail_text,
                "label": rel_type
            })
        
        samples.append({
            "text": text,
            "entities": entities,
            "relations": relations
        })
    
    return samples


def generate_ja_samples(count: int) -> List[Dict]:
    """Generate Japanese samples."""
    templates = [
        # founder_of + founded_in + located_in
        ("{person}は{year}に{location}で{org}を設立した。",
         [("person", "{person}"), ("date", "{year}"), ("location", "{location}"), ("organisation", "{org}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{year}", "founded_in"), ("{org}", "{location}", "located_in")]),
        
        ("{org}は{person}によって{year}に創業された。",
         [("organisation", "{org}"), ("person", "{person}"), ("date", "{year}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{year}", "founded_in")]),
        
        ("{org}は{person}が創立し、本社は{location}にある。",
         [("organisation", "{org}"), ("person", "{person}"), ("location", "{location}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{location}", "located_in")]),
        
        # ceo_of
        ("{person}は{org}のCEOである。",
         [("person", "{person}"), ("organisation", "{org}")],
         [("{person}", "{org}", "ceo_of")]),
        
        ("{person}は{org}の社長を務めている。",
         [("person", "{person}"), ("organisation", "{org}")],
         [("{person}", "{org}", "ceo_of")]),
        
        # developed
        ("{org}は{product}を開発した。",
         [("organisation", "{org}"), ("product", "{product}")],
         [("{org}", "{product}", "developed")]),
        
        ("{product}は{org}によって{year}に発売された。",
         [("product", "{product}"), ("organisation", "{org}"), ("date", "{year}")],
         [("{org}", "{product}", "developed"), ("{product}", "{year}", "released_in")]),
        
        # located_in
        ("{org}の本社は{location}にある。",
         [("organisation", "{org}"), ("location", "{location}")],
         [("{org}", "{location}", "located_in")]),
        
        # Complex sentences
        ("{person}と{person2}は{year}に{org}を共同設立した。",
         [("person", "{person}"), ("person", "{person2}"), ("date", "{year}"), ("organisation", "{org}")],
         [("{person}", "{org}", "founder_of"), ("{person2}", "{org}", "founder_of"), ("{org}", "{year}", "founded_in")]),
        
        ("{org}は{year}に設立され、{person}が創業者である。本社は{location}にある。",
         [("organisation", "{org}"), ("date", "{year}"), ("person", "{person}"), ("location", "{location}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{year}", "founded_in"), ("{org}", "{location}", "located_in")]),
        
        ("{org}は{person}が{year}に{location}で創業し、{product}を開発した。",
         [("organisation", "{org}"), ("person", "{person}"), ("date", "{year}"), ("location", "{location}"), ("product", "{product}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{year}", "founded_in"), ("{org}", "{location}", "located_in"), ("{org}", "{product}", "developed")]),
        
        ("{person}は{org}の創業者兼CEOである。",
         [("person", "{person}"), ("organisation", "{org}")],
         [("{person}", "{org}", "founder_of"), ("{person}", "{org}", "ceo_of")]),
    ]
    
    samples = []
    for _ in range(count):
        template, entity_templates, relation_templates = random.choice(templates)
        
        replacements = {
            "{person}": random.choice(JA_PERSONS),
            "{person2}": random.choice(JA_PERSONS),
            "{org}": random.choice(JA_ORGS),
            "{location}": random.choice(JA_LOCATIONS),
            "{product}": random.choice(JA_PRODUCTS),
            "{year}": str(random.choice(YEARS)) + "年",
        }
        
        while replacements["{person}"] == replacements["{person2}"]:
            replacements["{person2}"] = random.choice(JA_PERSONS)
        
        text = template
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, value)
        
        entities = []
        for label, placeholder in entity_templates:
            entity_text = replacements[placeholder]
            start = text.find(entity_text)
            if start != -1:
                if not any(e["text"] == entity_text and e["start"] == start for e in entities):
                    entities.append({
                        "start": start,
                        "end": start + len(entity_text),
                        "label": label,
                        "text": entity_text
                    })
        
        relations = []
        for head_ph, tail_ph, rel_type in relation_templates:
            head_text = replacements[head_ph]
            tail_text = replacements[tail_ph]
            relations.append({
                "head": head_text,
                "tail": tail_text,
                "label": rel_type
            })
        
        samples.append({
            "text": text,
            "entities": entities,
            "relations": relations
        })
    
    return samples


def generate_ko_samples(count: int) -> List[Dict]:
    """Generate Korean samples."""
    templates = [
        # founder_of + founded_in + located_in
        ("{person}은(는) {year}년 {location}에서 {org}을(를) 설립했다.",
         [("person", "{person}"), ("date", "{year}년"), ("location", "{location}"), ("organisation", "{org}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{year}년", "founded_in"), ("{org}", "{location}", "located_in")]),
        
        ("{org}은(는) {person}이(가) {year}년에 창업했다.",
         [("organisation", "{org}"), ("person", "{person}"), ("date", "{year}년")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{year}년", "founded_in")]),
        
        ("{org}은(는) {person}이(가) 설립했으며 본사는 {location}에 있다.",
         [("organisation", "{org}"), ("person", "{person}"), ("location", "{location}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{location}", "located_in")]),
        
        # ceo_of
        ("{person}은(는) {org}의 CEO이다.",
         [("person", "{person}"), ("organisation", "{org}")],
         [("{person}", "{org}", "ceo_of")]),
        
        ("{person}은(는) {org}의 대표이사를 맡고 있다.",
         [("person", "{person}"), ("organisation", "{org}")],
         [("{person}", "{org}", "ceo_of")]),
        
        # developed
        ("{org}은(는) {product}을(를) 개발했다.",
         [("organisation", "{org}"), ("product", "{product}")],
         [("{org}", "{product}", "developed")]),
        
        ("{product}은(는) {org}에서 {year}년에 출시되었다.",
         [("product", "{product}"), ("organisation", "{org}"), ("date", "{year}년")],
         [("{org}", "{product}", "developed"), ("{product}", "{year}년", "released_in")]),
        
        # located_in
        ("{org}의 본사는 {location}에 위치해 있다.",
         [("organisation", "{org}"), ("location", "{location}")],
         [("{org}", "{location}", "located_in")]),
        
        # Complex sentences
        ("{person}과(와) {person2}은(는) {year}년에 {org}을(를) 공동 설립했다.",
         [("person", "{person}"), ("person", "{person2}"), ("date", "{year}년"), ("organisation", "{org}")],
         [("{person}", "{org}", "founder_of"), ("{person2}", "{org}", "founder_of"), ("{org}", "{year}년", "founded_in")]),
        
        ("{org}은(는) {year}년에 설립되었으며, {person}이(가) 창업자이다. 본사는 {location}에 있다.",
         [("organisation", "{org}"), ("date", "{year}년"), ("person", "{person}"), ("location", "{location}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{year}년", "founded_in"), ("{org}", "{location}", "located_in")]),
        
        ("{person}은(는) {org}의 창업자이자 CEO이다.",
         [("person", "{person}"), ("organisation", "{org}")],
         [("{person}", "{org}", "founder_of"), ("{person}", "{org}", "ceo_of")]),
    ]
    
    samples = []
    for _ in range(count):
        template, entity_templates, relation_templates = random.choice(templates)
        
        year_val = str(random.choice(YEARS))
        replacements = {
            "{person}": random.choice(KO_PERSONS),
            "{person2}": random.choice(KO_PERSONS),
            "{org}": random.choice(KO_ORGS),
            "{location}": random.choice(KO_LOCATIONS),
            "{product}": random.choice(KO_PRODUCTS),
            "{year}년": year_val + "년",
        }
        
        while replacements["{person}"] == replacements["{person2}"]:
            replacements["{person2}"] = random.choice(KO_PERSONS)
        
        text = template
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, value)
        
        entities = []
        for label, placeholder in entity_templates:
            entity_text = replacements[placeholder]
            start = text.find(entity_text)
            if start != -1:
                if not any(e["text"] == entity_text and e["start"] == start for e in entities):
                    entities.append({
                        "start": start,
                        "end": start + len(entity_text),
                        "label": label,
                        "text": entity_text
                    })
        
        relations = []
        for head_ph, tail_ph, rel_type in relation_templates:
            head_text = replacements[head_ph]
            tail_text = replacements[tail_ph]
            relations.append({
                "head": head_text,
                "tail": tail_text,
                "label": rel_type
            })
        
        samples.append({
            "text": text,
            "entities": entities,
            "relations": relations
        })
    
    return samples


def generate_th_samples(count: int) -> List[Dict]:
    """Generate Thai samples."""
    templates = [
        # founder_of + founded_in + located_in
        ("{person} ก่อตั้ง {org} ในปี {year} ที่{location}",
         [("person", "{person}"), ("organisation", "{org}"), ("date", "{year}"), ("location", "{location}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{year}", "founded_in"), ("{org}", "{location}", "located_in")]),
        
        ("{org} ก่อตั้งโดย {person} ในปี {year}",
         [("organisation", "{org}"), ("person", "{person}"), ("date", "{year}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{year}", "founded_in")]),
        
        ("{org} ก่อตั้งโดย {person} มีสำนักงานใหญ่ที่{location}",
         [("organisation", "{org}"), ("person", "{person}"), ("location", "{location}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{location}", "located_in")]),
        
        # ceo_of
        ("{person} เป็นซีอีโอของ {org}",
         [("person", "{person}"), ("organisation", "{org}")],
         [("{person}", "{org}", "ceo_of")]),
        
        ("{person} ดำรงตำแหน่งประธานเจ้าหน้าที่บริหารของ {org}",
         [("person", "{person}"), ("organisation", "{org}")],
         [("{person}", "{org}", "ceo_of")]),
        
        # developed
        ("{org} พัฒนา {product}",
         [("organisation", "{org}"), ("product", "{product}")],
         [("{org}", "{product}", "developed")]),
        
        ("{product} พัฒนาโดย {org} ในปี {year}",
         [("product", "{product}"), ("organisation", "{org}"), ("date", "{year}")],
         [("{org}", "{product}", "developed"), ("{product}", "{year}", "released_in")]),
        
        # located_in
        ("{org} มีสำนักงานใหญ่ที่{location}",
         [("organisation", "{org}"), ("location", "{location}")],
         [("{org}", "{location}", "located_in")]),
        
        # Complex sentences
        ("{org} ก่อตั้งในปี {year} โดย {person} ที่{location}",
         [("organisation", "{org}"), ("date", "{year}"), ("person", "{person}"), ("location", "{location}")],
         [("{person}", "{org}", "founder_of"), ("{org}", "{year}", "founded_in"), ("{org}", "{location}", "located_in")]),
        
        ("{person} เป็นผู้ก่อตั้งและซีอีโอของ {org}",
         [("person", "{person}"), ("organisation", "{org}")],
         [("{person}", "{org}", "founder_of"), ("{person}", "{org}", "ceo_of")]),
    ]
    
    samples = []
    for _ in range(count):
        template, entity_templates, relation_templates = random.choice(templates)
        
        replacements = {
            "{person}": random.choice(TH_PERSONS),
            "{org}": random.choice(TH_ORGS),
            "{location}": random.choice(TH_LOCATIONS),
            "{product}": random.choice(TH_PRODUCTS),
            "{year}": str(random.choice(YEARS)),
        }
        
        text = template
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, value)
        
        entities = []
        for label, placeholder in entity_templates:
            entity_text = replacements[placeholder]
            start = text.find(entity_text)
            if start != -1:
                if not any(e["text"] == entity_text and e["start"] == start for e in entities):
                    entities.append({
                        "start": start,
                        "end": start + len(entity_text),
                        "label": label,
                        "text": entity_text
                    })
        
        relations = []
        for head_ph, tail_ph, rel_type in relation_templates:
            head_text = replacements[head_ph]
            tail_text = replacements[tail_ph]
            relations.append({
                "head": head_text,
                "tail": tail_text,
                "label": rel_type
            })
        
        samples.append({
            "text": text,
            "entities": entities,
            "relations": relations
        })
    
    return samples


def generate_dataset(target_count: int = 5000) -> List[Dict]:
    """Generate a balanced multilingual dataset."""
    
    # Distribution: 30% EN, 25% ZH, 20% JA, 15% KO, 10% TH
    en_count = int(target_count * 0.30)
    zh_count = int(target_count * 0.25)
    ja_count = int(target_count * 0.20)
    ko_count = int(target_count * 0.15)
    th_count = int(target_count * 0.10)
    
    print(f"Generating {en_count} English samples...")
    en_samples = generate_en_samples(en_count)
    
    print(f"Generating {zh_count} Chinese samples...")
    zh_samples = generate_zh_samples(zh_count)
    
    print(f"Generating {ja_count} Japanese samples...")
    ja_samples = generate_ja_samples(ja_count)
    
    print(f"Generating {ko_count} Korean samples...")
    ko_samples = generate_ko_samples(ko_count)
    
    print(f"Generating {th_count} Thai samples...")
    th_samples = generate_th_samples(th_count)
    
    # Combine and shuffle
    all_samples = en_samples + zh_samples + ja_samples + ko_samples + th_samples
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
    
    print("\nEntity Type Distribution:")
    for label, count in sorted(entity_types.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count} ({count/total_entities*100:.1f}%)")
    
    print("\nRelation Type Distribution:")
    for label, count in sorted(relation_types.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count} ({count/total_relations*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate large multilingual NER+RE dataset")
    parser.add_argument("--count", type=int, default=5000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="multilingual_data_v3_5000.json", help="Output file name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print(f"Generating {args.count} multilingual samples...")
    samples = generate_dataset(args.count)
    
    print_statistics(samples)
    
    output_path = f"/data/tcustpg18/NERRE/NERRE/dataset/{args.output}"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"\nDataset saved to: {output_path}")
