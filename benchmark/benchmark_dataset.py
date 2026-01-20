"""
Standard Benchmark Dataset for NERRE
=====================================
Large-scale test dataset aligned with NERRE's relation labels.

NERRE Relation Labels (from config.json):
- ceo_of: person -> organisation
- creator_of: person/organisation -> product/programlang
- developed: person/organisation -> product/programlang
- founded_in: organisation/product -> date/location
- founder_of: person -> organisation
- located_in: organisation/person -> location
- released_in: product/programlang -> date/location

NERRE Entity Labels:
- person, organisation, location, date, product, programlang
"""

from typing import List, Dict


def get_benchmark_dataset() -> List[Dict]:
    """
    Returns a comprehensive benchmark dataset with 100+ samples.
    All relations use NERRE's exact label format.
    """
    
    dataset = []
    
    # ============================================================
    # English - Tech Companies & Founders (30 samples)
    # ============================================================
    
    # 1. Microsoft
    dataset.append({
        "text": "Bill Gates founded Microsoft in 1975.",
        "entities": [
            {"text": "Bill Gates", "label": "person", "start": 0, "end": 10},
            {"text": "Microsoft", "label": "organisation", "start": 19, "end": 28},
            {"text": "1975", "label": "date", "start": 32, "end": 36},
        ],
        "relations": [
            {"head": "Bill Gates", "tail": "Microsoft", "relation": "founder_of"},
            {"head": "Microsoft", "tail": "1975", "relation": "founded_in"}
        ]
    })
    
    # 2. Apple
    dataset.append({
        "text": "Steve Jobs founded Apple in 1976 in California.",
        "entities": [
            {"text": "Steve Jobs", "label": "person", "start": 0, "end": 10},
            {"text": "Apple", "label": "organisation", "start": 19, "end": 24},
            {"text": "1976", "label": "date", "start": 28, "end": 32},
            {"text": "California", "label": "location", "start": 36, "end": 46},
        ],
        "relations": [
            {"head": "Steve Jobs", "tail": "Apple", "relation": "founder_of"},
            {"head": "Apple", "tail": "1976", "relation": "founded_in"},
            {"head": "Apple", "tail": "California", "relation": "located_in"}
        ]
    })
    
    # 3. Tesla
    dataset.append({
        "text": "Elon Musk is the CEO of Tesla.",
        "entities": [
            {"text": "Elon Musk", "label": "person", "start": 0, "end": 9},
            {"text": "Tesla", "label": "organisation", "start": 25, "end": 30},
        ],
        "relations": [
            {"head": "Elon Musk", "tail": "Tesla", "relation": "ceo_of"}
        ]
    })
    
    # 4. SpaceX
    dataset.append({
        "text": "Elon Musk founded SpaceX in 2002.",
        "entities": [
            {"text": "Elon Musk", "label": "person", "start": 0, "end": 9},
            {"text": "SpaceX", "label": "organisation", "start": 18, "end": 24},
            {"text": "2002", "label": "date", "start": 28, "end": 32},
        ],
        "relations": [
            {"head": "Elon Musk", "tail": "SpaceX", "relation": "founder_of"},
            {"head": "SpaceX", "tail": "2002", "relation": "founded_in"}
        ]
    })
    
    # 5. Meta/Facebook
    dataset.append({
        "text": "Mark Zuckerberg is the CEO of Meta.",
        "entities": [
            {"text": "Mark Zuckerberg", "label": "person", "start": 0, "end": 15},
            {"text": "Meta", "label": "organisation", "start": 30, "end": 34},
        ],
        "relations": [
            {"head": "Mark Zuckerberg", "tail": "Meta", "relation": "ceo_of"}
        ]
    })
    
    # 6. Facebook founding
    dataset.append({
        "text": "Mark Zuckerberg founded Facebook in 2004.",
        "entities": [
            {"text": "Mark Zuckerberg", "label": "person", "start": 0, "end": 15},
            {"text": "Facebook", "label": "organisation", "start": 24, "end": 32},
            {"text": "2004", "label": "date", "start": 36, "end": 40},
        ],
        "relations": [
            {"head": "Mark Zuckerberg", "tail": "Facebook", "relation": "founder_of"},
            {"head": "Facebook", "tail": "2004", "relation": "founded_in"}
        ]
    })
    
    # 7. Amazon
    dataset.append({
        "text": "Jeff Bezos founded Amazon in 1994 in Seattle.",
        "entities": [
            {"text": "Jeff Bezos", "label": "person", "start": 0, "end": 10},
            {"text": "Amazon", "label": "organisation", "start": 19, "end": 25},
            {"text": "1994", "label": "date", "start": 29, "end": 33},
            {"text": "Seattle", "label": "location", "start": 37, "end": 44},
        ],
        "relations": [
            {"head": "Jeff Bezos", "tail": "Amazon", "relation": "founder_of"},
            {"head": "Amazon", "tail": "1994", "relation": "founded_in"},
            {"head": "Amazon", "tail": "Seattle", "relation": "located_in"}
        ]
    })
    
    # 8. Google
    dataset.append({
        "text": "Larry Page and Sergey Brin founded Google in 1998.",
        "entities": [
            {"text": "Larry Page", "label": "person", "start": 0, "end": 10},
            {"text": "Sergey Brin", "label": "person", "start": 15, "end": 26},
            {"text": "Google", "label": "organisation", "start": 35, "end": 41},
            {"text": "1998", "label": "date", "start": 45, "end": 49},
        ],
        "relations": [
            {"head": "Larry Page", "tail": "Google", "relation": "founder_of"},
            {"head": "Sergey Brin", "tail": "Google", "relation": "founder_of"},
            {"head": "Google", "tail": "1998", "relation": "founded_in"}
        ]
    })
    
    # 9. Sundar Pichai
    dataset.append({
        "text": "Sundar Pichai is the CEO of Google.",
        "entities": [
            {"text": "Sundar Pichai", "label": "person", "start": 0, "end": 13},
            {"text": "Google", "label": "organisation", "start": 28, "end": 34},
        ],
        "relations": [
            {"head": "Sundar Pichai", "tail": "Google", "relation": "ceo_of"}
        ]
    })
    
    # 10. NVIDIA
    dataset.append({
        "text": "Jensen Huang founded NVIDIA in 1993.",
        "entities": [
            {"text": "Jensen Huang", "label": "person", "start": 0, "end": 12},
            {"text": "NVIDIA", "label": "organisation", "start": 21, "end": 27},
            {"text": "1993", "label": "date", "start": 31, "end": 35},
        ],
        "relations": [
            {"head": "Jensen Huang", "tail": "NVIDIA", "relation": "founder_of"},
            {"head": "NVIDIA", "tail": "1993", "relation": "founded_in"}
        ]
    })
    
    # 11. Jensen Huang CEO
    dataset.append({
        "text": "Jensen Huang is the CEO of NVIDIA.",
        "entities": [
            {"text": "Jensen Huang", "label": "person", "start": 0, "end": 12},
            {"text": "NVIDIA", "label": "organisation", "start": 27, "end": 33},
        ],
        "relations": [
            {"head": "Jensen Huang", "tail": "NVIDIA", "relation": "ceo_of"}
        ]
    })
    
    # 12. Tim Cook
    dataset.append({
        "text": "Tim Cook is the CEO of Apple.",
        "entities": [
            {"text": "Tim Cook", "label": "person", "start": 0, "end": 8},
            {"text": "Apple", "label": "organisation", "start": 23, "end": 28},
        ],
        "relations": [
            {"head": "Tim Cook", "tail": "Apple", "relation": "ceo_of"}
        ]
    })
    
    # 13. Satya Nadella
    dataset.append({
        "text": "Satya Nadella is the CEO of Microsoft.",
        "entities": [
            {"text": "Satya Nadella", "label": "person", "start": 0, "end": 13},
            {"text": "Microsoft", "label": "organisation", "start": 28, "end": 37},
        ],
        "relations": [
            {"head": "Satya Nadella", "tail": "Microsoft", "relation": "ceo_of"}
        ]
    })
    
    # 14. OpenAI
    dataset.append({
        "text": "Sam Altman is the CEO of OpenAI.",
        "entities": [
            {"text": "Sam Altman", "label": "person", "start": 0, "end": 10},
            {"text": "OpenAI", "label": "organisation", "start": 25, "end": 31},
        ],
        "relations": [
            {"head": "Sam Altman", "tail": "OpenAI", "relation": "ceo_of"}
        ]
    })
    
    # 15. OpenAI founding
    dataset.append({
        "text": "OpenAI was founded in 2015 in San Francisco.",
        "entities": [
            {"text": "OpenAI", "label": "organisation", "start": 0, "end": 6},
            {"text": "2015", "label": "date", "start": 22, "end": 26},
            {"text": "San Francisco", "label": "location", "start": 30, "end": 43},
        ],
        "relations": [
            {"head": "OpenAI", "tail": "2015", "relation": "founded_in"},
            {"head": "OpenAI", "tail": "San Francisco", "relation": "located_in"}
        ]
    })
    
    # 16. Anthropic
    dataset.append({
        "text": "Dario Amodei is the CEO of Anthropic.",
        "entities": [
            {"text": "Dario Amodei", "label": "person", "start": 0, "end": 12},
            {"text": "Anthropic", "label": "organisation", "start": 27, "end": 36},
        ],
        "relations": [
            {"head": "Dario Amodei", "tail": "Anthropic", "relation": "ceo_of"}
        ]
    })
    
    # 17. Anthropic founding
    dataset.append({
        "text": "Dario Amodei founded Anthropic in 2021.",
        "entities": [
            {"text": "Dario Amodei", "label": "person", "start": 0, "end": 12},
            {"text": "Anthropic", "label": "organisation", "start": 21, "end": 30},
            {"text": "2021", "label": "date", "start": 34, "end": 38},
        ],
        "relations": [
            {"head": "Dario Amodei", "tail": "Anthropic", "relation": "founder_of"},
            {"head": "Anthropic", "tail": "2021", "relation": "founded_in"}
        ]
    })
    
    # 18. Twitter/X
    dataset.append({
        "text": "Jack Dorsey founded Twitter in 2006.",
        "entities": [
            {"text": "Jack Dorsey", "label": "person", "start": 0, "end": 11},
            {"text": "Twitter", "label": "organisation", "start": 20, "end": 27},
            {"text": "2006", "label": "date", "start": 31, "end": 35},
        ],
        "relations": [
            {"head": "Jack Dorsey", "tail": "Twitter", "relation": "founder_of"},
            {"head": "Twitter", "tail": "2006", "relation": "founded_in"}
        ]
    })
    
    # 19. Netflix
    dataset.append({
        "text": "Reed Hastings founded Netflix in 1997.",
        "entities": [
            {"text": "Reed Hastings", "label": "person", "start": 0, "end": 13},
            {"text": "Netflix", "label": "organisation", "start": 22, "end": 29},
            {"text": "1997", "label": "date", "start": 33, "end": 37},
        ],
        "relations": [
            {"head": "Reed Hastings", "tail": "Netflix", "relation": "founder_of"},
            {"head": "Netflix", "tail": "1997", "relation": "founded_in"}
        ]
    })
    
    # 20. Alibaba
    dataset.append({
        "text": "Jack Ma founded Alibaba in 1999 in Hangzhou.",
        "entities": [
            {"text": "Jack Ma", "label": "person", "start": 0, "end": 7},
            {"text": "Alibaba", "label": "organisation", "start": 16, "end": 23},
            {"text": "1999", "label": "date", "start": 27, "end": 31},
            {"text": "Hangzhou", "label": "location", "start": 35, "end": 43},
        ],
        "relations": [
            {"head": "Jack Ma", "tail": "Alibaba", "relation": "founder_of"},
            {"head": "Alibaba", "tail": "1999", "relation": "founded_in"},
            {"head": "Alibaba", "tail": "Hangzhou", "relation": "located_in"}
        ]
    })
    
    # ============================================================
    # Products & Programming Languages (20 samples)
    # ============================================================
    
    # 21. Python
    dataset.append({
        "text": "Guido van Rossum created Python in 1991.",
        "entities": [
            {"text": "Guido van Rossum", "label": "person", "start": 0, "end": 16},
            {"text": "Python", "label": "programlang", "start": 25, "end": 31},
            {"text": "1991", "label": "date", "start": 35, "end": 39},
        ],
        "relations": [
            {"head": "Guido van Rossum", "tail": "Python", "relation": "creator_of"},
            {"head": "Python", "tail": "1991", "relation": "released_in"}
        ]
    })
    
    # 22. Java
    dataset.append({
        "text": "James Gosling created Java at Sun Microsystems in 1995.",
        "entities": [
            {"text": "James Gosling", "label": "person", "start": 0, "end": 13},
            {"text": "Java", "label": "programlang", "start": 22, "end": 26},
            {"text": "Sun Microsystems", "label": "organisation", "start": 30, "end": 46},
            {"text": "1995", "label": "date", "start": 50, "end": 54},
        ],
        "relations": [
            {"head": "James Gosling", "tail": "Java", "relation": "creator_of"},
            {"head": "Java", "tail": "1995", "relation": "released_in"}
        ]
    })
    
    # 23. JavaScript
    dataset.append({
        "text": "Brendan Eich created JavaScript in 1995.",
        "entities": [
            {"text": "Brendan Eich", "label": "person", "start": 0, "end": 12},
            {"text": "JavaScript", "label": "programlang", "start": 21, "end": 31},
            {"text": "1995", "label": "date", "start": 35, "end": 39},
        ],
        "relations": [
            {"head": "Brendan Eich", "tail": "JavaScript", "relation": "creator_of"},
            {"head": "JavaScript", "tail": "1995", "relation": "released_in"}
        ]
    })
    
    # 24. C++
    dataset.append({
        "text": "Bjarne Stroustrup created C++ in 1983.",
        "entities": [
            {"text": "Bjarne Stroustrup", "label": "person", "start": 0, "end": 17},
            {"text": "C++", "label": "programlang", "start": 26, "end": 29},
            {"text": "1983", "label": "date", "start": 33, "end": 37},
        ],
        "relations": [
            {"head": "Bjarne Stroustrup", "tail": "C++", "relation": "creator_of"},
            {"head": "C++", "tail": "1983", "relation": "released_in"}
        ]
    })
    
    # 25. Ruby
    dataset.append({
        "text": "Yukihiro Matsumoto created Ruby in 1995 in Japan.",
        "entities": [
            {"text": "Yukihiro Matsumoto", "label": "person", "start": 0, "end": 18},
            {"text": "Ruby", "label": "programlang", "start": 27, "end": 31},
            {"text": "1995", "label": "date", "start": 35, "end": 39},
            {"text": "Japan", "label": "location", "start": 43, "end": 48},
        ],
        "relations": [
            {"head": "Yukihiro Matsumoto", "tail": "Ruby", "relation": "creator_of"},
            {"head": "Ruby", "tail": "1995", "relation": "released_in"}
        ]
    })
    
    # 26. Go
    dataset.append({
        "text": "Google developed Go in 2009.",
        "entities": [
            {"text": "Google", "label": "organisation", "start": 0, "end": 6},
            {"text": "Go", "label": "programlang", "start": 17, "end": 19},
            {"text": "2009", "label": "date", "start": 23, "end": 27},
        ],
        "relations": [
            {"head": "Google", "tail": "Go", "relation": "developed"},
            {"head": "Go", "tail": "2009", "relation": "released_in"}
        ]
    })
    
    # 27. Rust
    dataset.append({
        "text": "Mozilla developed Rust in 2010.",
        "entities": [
            {"text": "Mozilla", "label": "organisation", "start": 0, "end": 7},
            {"text": "Rust", "label": "programlang", "start": 18, "end": 22},
            {"text": "2010", "label": "date", "start": 26, "end": 30},
        ],
        "relations": [
            {"head": "Mozilla", "tail": "Rust", "relation": "developed"},
            {"head": "Rust", "tail": "2010", "relation": "released_in"}
        ]
    })
    
    # 28. Swift
    dataset.append({
        "text": "Apple developed Swift in 2014.",
        "entities": [
            {"text": "Apple", "label": "organisation", "start": 0, "end": 5},
            {"text": "Swift", "label": "programlang", "start": 16, "end": 21},
            {"text": "2014", "label": "date", "start": 25, "end": 29},
        ],
        "relations": [
            {"head": "Apple", "tail": "Swift", "relation": "developed"},
            {"head": "Swift", "tail": "2014", "relation": "released_in"}
        ]
    })
    
    # 29. Kotlin
    dataset.append({
        "text": "JetBrains developed Kotlin in 2011.",
        "entities": [
            {"text": "JetBrains", "label": "organisation", "start": 0, "end": 9},
            {"text": "Kotlin", "label": "programlang", "start": 20, "end": 26},
            {"text": "2011", "label": "date", "start": 30, "end": 34},
        ],
        "relations": [
            {"head": "JetBrains", "tail": "Kotlin", "relation": "developed"},
            {"head": "Kotlin", "tail": "2011", "relation": "released_in"}
        ]
    })
    
    # 30. TypeScript
    dataset.append({
        "text": "Microsoft developed TypeScript in 2012.",
        "entities": [
            {"text": "Microsoft", "label": "organisation", "start": 0, "end": 9},
            {"text": "TypeScript", "label": "programlang", "start": 20, "end": 30},
            {"text": "2012", "label": "date", "start": 34, "end": 38},
        ],
        "relations": [
            {"head": "Microsoft", "tail": "TypeScript", "relation": "developed"},
            {"head": "TypeScript", "tail": "2012", "relation": "released_in"}
        ]
    })
    
    # 31. iPhone
    dataset.append({
        "text": "Apple developed the iPhone in 2007.",
        "entities": [
            {"text": "Apple", "label": "organisation", "start": 0, "end": 5},
            {"text": "iPhone", "label": "product", "start": 20, "end": 26},
            {"text": "2007", "label": "date", "start": 30, "end": 34},
        ],
        "relations": [
            {"head": "Apple", "tail": "iPhone", "relation": "developed"},
            {"head": "iPhone", "tail": "2007", "relation": "released_in"}
        ]
    })
    
    # 32. ChatGPT
    dataset.append({
        "text": "OpenAI developed ChatGPT in 2022.",
        "entities": [
            {"text": "OpenAI", "label": "organisation", "start": 0, "end": 6},
            {"text": "ChatGPT", "label": "product", "start": 17, "end": 24},
            {"text": "2022", "label": "date", "start": 28, "end": 32},
        ],
        "relations": [
            {"head": "OpenAI", "tail": "ChatGPT", "relation": "developed"},
            {"head": "ChatGPT", "tail": "2022", "relation": "released_in"}
        ]
    })
    
    # 33. Windows
    dataset.append({
        "text": "Microsoft developed Windows in 1985.",
        "entities": [
            {"text": "Microsoft", "label": "organisation", "start": 0, "end": 9},
            {"text": "Windows", "label": "product", "start": 20, "end": 27},
            {"text": "1985", "label": "date", "start": 31, "end": 35},
        ],
        "relations": [
            {"head": "Microsoft", "tail": "Windows", "relation": "developed"},
            {"head": "Windows", "tail": "1985", "relation": "released_in"}
        ]
    })
    
    # 34. Android
    dataset.append({
        "text": "Google developed Android in 2008.",
        "entities": [
            {"text": "Google", "label": "organisation", "start": 0, "end": 6},
            {"text": "Android", "label": "product", "start": 17, "end": 24},
            {"text": "2008", "label": "date", "start": 28, "end": 32},
        ],
        "relations": [
            {"head": "Google", "tail": "Android", "relation": "developed"},
            {"head": "Android", "tail": "2008", "relation": "released_in"}
        ]
    })
    
    # 35. Linux
    dataset.append({
        "text": "Linus Torvalds created Linux in 1991.",
        "entities": [
            {"text": "Linus Torvalds", "label": "person", "start": 0, "end": 14},
            {"text": "Linux", "label": "product", "start": 23, "end": 28},
            {"text": "1991", "label": "date", "start": 32, "end": 36},
        ],
        "relations": [
            {"head": "Linus Torvalds", "tail": "Linux", "relation": "creator_of"},
            {"head": "Linux", "tail": "1991", "relation": "released_in"}
        ]
    })
    
    # 36. Git
    dataset.append({
        "text": "Linus Torvalds created Git in 2005.",
        "entities": [
            {"text": "Linus Torvalds", "label": "person", "start": 0, "end": 14},
            {"text": "Git", "label": "product", "start": 23, "end": 26},
            {"text": "2005", "label": "date", "start": 30, "end": 34},
        ],
        "relations": [
            {"head": "Linus Torvalds", "tail": "Git", "relation": "creator_of"},
            {"head": "Git", "tail": "2005", "relation": "released_in"}
        ]
    })
    
    # 37. PyTorch
    dataset.append({
        "text": "Meta developed PyTorch in 2016.",
        "entities": [
            {"text": "Meta", "label": "organisation", "start": 0, "end": 4},
            {"text": "PyTorch", "label": "product", "start": 15, "end": 22},
            {"text": "2016", "label": "date", "start": 26, "end": 30},
        ],
        "relations": [
            {"head": "Meta", "tail": "PyTorch", "relation": "developed"},
            {"head": "PyTorch", "tail": "2016", "relation": "released_in"}
        ]
    })
    
    # 38. TensorFlow
    dataset.append({
        "text": "Google developed TensorFlow in 2015.",
        "entities": [
            {"text": "Google", "label": "organisation", "start": 0, "end": 6},
            {"text": "TensorFlow", "label": "product", "start": 17, "end": 27},
            {"text": "2015", "label": "date", "start": 31, "end": 35},
        ],
        "relations": [
            {"head": "Google", "tail": "TensorFlow", "relation": "developed"},
            {"head": "TensorFlow", "tail": "2015", "relation": "released_in"}
        ]
    })
    
    # 39. Claude
    dataset.append({
        "text": "Anthropic developed Claude in 2023.",
        "entities": [
            {"text": "Anthropic", "label": "organisation", "start": 0, "end": 9},
            {"text": "Claude", "label": "product", "start": 20, "end": 26},
            {"text": "2023", "label": "date", "start": 30, "end": 34},
        ],
        "relations": [
            {"head": "Anthropic", "tail": "Claude", "relation": "developed"},
            {"head": "Claude", "tail": "2023", "relation": "released_in"}
        ]
    })
    
    # 40. GPT-4
    dataset.append({
        "text": "OpenAI developed GPT-4 in 2023.",
        "entities": [
            {"text": "OpenAI", "label": "organisation", "start": 0, "end": 6},
            {"text": "GPT-4", "label": "product", "start": 17, "end": 22},
            {"text": "2023", "label": "date", "start": 26, "end": 30},
        ],
        "relations": [
            {"head": "OpenAI", "tail": "GPT-4", "relation": "developed"},
            {"head": "GPT-4", "tail": "2023", "relation": "released_in"}
        ]
    })
    
    # ============================================================
    # Company Locations (15 samples)
    # ============================================================
    
    # 41. Apple HQ
    dataset.append({
        "text": "Apple is located in Cupertino, California.",
        "entities": [
            {"text": "Apple", "label": "organisation", "start": 0, "end": 5},
            {"text": "Cupertino", "label": "location", "start": 20, "end": 29},
            {"text": "California", "label": "location", "start": 31, "end": 41},
        ],
        "relations": [
            {"head": "Apple", "tail": "Cupertino", "relation": "located_in"}
        ]
    })
    
    # 42. Google HQ
    dataset.append({
        "text": "Google is located in Mountain View.",
        "entities": [
            {"text": "Google", "label": "organisation", "start": 0, "end": 6},
            {"text": "Mountain View", "label": "location", "start": 21, "end": 34},
        ],
        "relations": [
            {"head": "Google", "tail": "Mountain View", "relation": "located_in"}
        ]
    })
    
    # 43. Microsoft HQ
    dataset.append({
        "text": "Microsoft is located in Redmond, Washington.",
        "entities": [
            {"text": "Microsoft", "label": "organisation", "start": 0, "end": 9},
            {"text": "Redmond", "label": "location", "start": 24, "end": 31},
            {"text": "Washington", "label": "location", "start": 33, "end": 43},
        ],
        "relations": [
            {"head": "Microsoft", "tail": "Redmond", "relation": "located_in"}
        ]
    })
    
    # 44. Amazon HQ
    dataset.append({
        "text": "Amazon is located in Seattle.",
        "entities": [
            {"text": "Amazon", "label": "organisation", "start": 0, "end": 6},
            {"text": "Seattle", "label": "location", "start": 21, "end": 28},
        ],
        "relations": [
            {"head": "Amazon", "tail": "Seattle", "relation": "located_in"}
        ]
    })
    
    # 45. Tesla HQ
    dataset.append({
        "text": "Tesla is located in Austin, Texas.",
        "entities": [
            {"text": "Tesla", "label": "organisation", "start": 0, "end": 5},
            {"text": "Austin", "label": "location", "start": 20, "end": 26},
            {"text": "Texas", "label": "location", "start": 28, "end": 33},
        ],
        "relations": [
            {"head": "Tesla", "tail": "Austin", "relation": "located_in"}
        ]
    })
    
    # 46. NVIDIA HQ
    dataset.append({
        "text": "NVIDIA is located in Santa Clara.",
        "entities": [
            {"text": "NVIDIA", "label": "organisation", "start": 0, "end": 6},
            {"text": "Santa Clara", "label": "location", "start": 21, "end": 32},
        ],
        "relations": [
            {"head": "NVIDIA", "tail": "Santa Clara", "relation": "located_in"}
        ]
    })
    
    # 47. Meta HQ
    dataset.append({
        "text": "Meta is located in Menlo Park.",
        "entities": [
            {"text": "Meta", "label": "organisation", "start": 0, "end": 4},
            {"text": "Menlo Park", "label": "location", "start": 19, "end": 29},
        ],
        "relations": [
            {"head": "Meta", "tail": "Menlo Park", "relation": "located_in"}
        ]
    })
    
    # 48. Netflix HQ
    dataset.append({
        "text": "Netflix is located in Los Gatos.",
        "entities": [
            {"text": "Netflix", "label": "organisation", "start": 0, "end": 7},
            {"text": "Los Gatos", "label": "location", "start": 22, "end": 31},
        ],
        "relations": [
            {"head": "Netflix", "tail": "Los Gatos", "relation": "located_in"}
        ]
    })
    
    # 49. Twitter HQ
    dataset.append({
        "text": "Twitter is located in San Francisco.",
        "entities": [
            {"text": "Twitter", "label": "organisation", "start": 0, "end": 7},
            {"text": "San Francisco", "label": "location", "start": 22, "end": 35},
        ],
        "relations": [
            {"head": "Twitter", "tail": "San Francisco", "relation": "located_in"}
        ]
    })
    
    # 50. Uber HQ
    dataset.append({
        "text": "Uber is located in San Francisco.",
        "entities": [
            {"text": "Uber", "label": "organisation", "start": 0, "end": 4},
            {"text": "San Francisco", "label": "location", "start": 19, "end": 32},
        ],
        "relations": [
            {"head": "Uber", "tail": "San Francisco", "relation": "located_in"}
        ]
    })
    
    # 51. Airbnb HQ
    dataset.append({
        "text": "Airbnb is located in San Francisco.",
        "entities": [
            {"text": "Airbnb", "label": "organisation", "start": 0, "end": 6},
            {"text": "San Francisco", "label": "location", "start": 21, "end": 34},
        ],
        "relations": [
            {"head": "Airbnb", "tail": "San Francisco", "relation": "located_in"}
        ]
    })
    
    # 52. Salesforce HQ
    dataset.append({
        "text": "Salesforce is located in San Francisco.",
        "entities": [
            {"text": "Salesforce", "label": "organisation", "start": 0, "end": 10},
            {"text": "San Francisco", "label": "location", "start": 25, "end": 38},
        ],
        "relations": [
            {"head": "Salesforce", "tail": "San Francisco", "relation": "located_in"}
        ]
    })
    
    # 53. Oracle HQ
    dataset.append({
        "text": "Oracle is located in Austin, Texas.",
        "entities": [
            {"text": "Oracle", "label": "organisation", "start": 0, "end": 6},
            {"text": "Austin", "label": "location", "start": 21, "end": 27},
            {"text": "Texas", "label": "location", "start": 29, "end": 34},
        ],
        "relations": [
            {"head": "Oracle", "tail": "Austin", "relation": "located_in"}
        ]
    })
    
    # 54. Adobe HQ
    dataset.append({
        "text": "Adobe is located in San Jose.",
        "entities": [
            {"text": "Adobe", "label": "organisation", "start": 0, "end": 5},
            {"text": "San Jose", "label": "location", "start": 20, "end": 28},
        ],
        "relations": [
            {"head": "Adobe", "tail": "San Jose", "relation": "located_in"}
        ]
    })
    
    # 55. Intel HQ
    dataset.append({
        "text": "Intel is located in Santa Clara.",
        "entities": [
            {"text": "Intel", "label": "organisation", "start": 0, "end": 5},
            {"text": "Santa Clara", "label": "location", "start": 20, "end": 31},
        ],
        "relations": [
            {"head": "Intel", "tail": "Santa Clara", "relation": "located_in"}
        ]
    })
    
    # ============================================================
    # Chinese Samples (15 samples)
    # ============================================================
    
    # 56. 阿里巴巴
    dataset.append({
        "text": "馬雲是阿里巴巴的創始人。",
        "entities": [
            {"text": "馬雲", "label": "person", "start": 0, "end": 2},
            {"text": "阿里巴巴", "label": "organisation", "start": 3, "end": 7},
        ],
        "relations": [
            {"head": "馬雲", "tail": "阿里巴巴", "relation": "founder_of"}
        ]
    })
    
    # 57. 騰訊
    dataset.append({
        "text": "馬化騰是騰訊的CEO。",
        "entities": [
            {"text": "馬化騰", "label": "person", "start": 0, "end": 3},
            {"text": "騰訊", "label": "organisation", "start": 4, "end": 6},
        ],
        "relations": [
            {"head": "馬化騰", "tail": "騰訊", "relation": "ceo_of"}
        ]
    })
    
    # 58. 騰訊創立
    dataset.append({
        "text": "馬化騰於1998年創立騰訊。",
        "entities": [
            {"text": "馬化騰", "label": "person", "start": 0, "end": 3},
            {"text": "1998年", "label": "date", "start": 4, "end": 9},
            {"text": "騰訊", "label": "organisation", "start": 11, "end": 13},
        ],
        "relations": [
            {"head": "馬化騰", "tail": "騰訊", "relation": "founder_of"},
            {"head": "騰訊", "tail": "1998年", "relation": "founded_in"}
        ]
    })
    
    # 59. 台積電
    dataset.append({
        "text": "張忠謀是台積電的創始人。",
        "entities": [
            {"text": "張忠謀", "label": "person", "start": 0, "end": 3},
            {"text": "台積電", "label": "organisation", "start": 4, "end": 7},
        ],
        "relations": [
            {"head": "張忠謀", "tail": "台積電", "relation": "founder_of"}
        ]
    })
    
    # 60. 台積電位置
    dataset.append({
        "text": "台積電位於新竹。",
        "entities": [
            {"text": "台積電", "label": "organisation", "start": 0, "end": 3},
            {"text": "新竹", "label": "location", "start": 5, "end": 7},
        ],
        "relations": [
            {"head": "台積電", "tail": "新竹", "relation": "located_in"}
        ]
    })
    
    # 61. 華為
    dataset.append({
        "text": "任正非是華為的創始人。",
        "entities": [
            {"text": "任正非", "label": "person", "start": 0, "end": 3},
            {"text": "華為", "label": "organisation", "start": 4, "end": 6},
        ],
        "relations": [
            {"head": "任正非", "tail": "華為", "relation": "founder_of"}
        ]
    })
    
    # 62. 小米
    dataset.append({
        "text": "雷軍是小米的CEO。",
        "entities": [
            {"text": "雷軍", "label": "person", "start": 0, "end": 2},
            {"text": "小米", "label": "organisation", "start": 3, "end": 5},
        ],
        "relations": [
            {"head": "雷軍", "tail": "小米", "relation": "ceo_of"}
        ]
    })
    
    # 63. 小米創立
    dataset.append({
        "text": "雷軍於2010年創立小米。",
        "entities": [
            {"text": "雷軍", "label": "person", "start": 0, "end": 2},
            {"text": "2010年", "label": "date", "start": 3, "end": 8},
            {"text": "小米", "label": "organisation", "start": 10, "end": 12},
        ],
        "relations": [
            {"head": "雷軍", "tail": "小米", "relation": "founder_of"},
            {"head": "小米", "tail": "2010年", "relation": "founded_in"}
        ]
    })
    
    # 64. 字節跳動
    dataset.append({
        "text": "張一鳴創立了字節跳動。",
        "entities": [
            {"text": "張一鳴", "label": "person", "start": 0, "end": 3},
            {"text": "字節跳動", "label": "organisation", "start": 6, "end": 10},
        ],
        "relations": [
            {"head": "張一鳴", "tail": "字節跳動", "relation": "founder_of"}
        ]
    })
    
    # 65. 京東
    dataset.append({
        "text": "劉強東是京東的創始人。",
        "entities": [
            {"text": "劉強東", "label": "person", "start": 0, "end": 3},
            {"text": "京東", "label": "organisation", "start": 4, "end": 6},
        ],
        "relations": [
            {"head": "劉強東", "tail": "京東", "relation": "founder_of"}
        ]
    })
    
    # 66. 百度
    dataset.append({
        "text": "李彥宏是百度的CEO。",
        "entities": [
            {"text": "李彥宏", "label": "person", "start": 0, "end": 3},
            {"text": "百度", "label": "organisation", "start": 4, "end": 6},
        ],
        "relations": [
            {"head": "李彥宏", "tail": "百度", "relation": "ceo_of"}
        ]
    })
    
    # 67. 百度創立
    dataset.append({
        "text": "李彥宏於2000年在北京創立百度。",
        "entities": [
            {"text": "李彥宏", "label": "person", "start": 0, "end": 3},
            {"text": "2000年", "label": "date", "start": 4, "end": 9},
            {"text": "北京", "label": "location", "start": 10, "end": 12},
            {"text": "百度", "label": "organisation", "start": 14, "end": 16},
        ],
        "relations": [
            {"head": "李彥宏", "tail": "百度", "relation": "founder_of"},
            {"head": "百度", "tail": "2000年", "relation": "founded_in"},
            {"head": "百度", "tail": "北京", "relation": "located_in"}
        ]
    })
    
    # 68. 微信
    dataset.append({
        "text": "騰訊開發了微信。",
        "entities": [
            {"text": "騰訊", "label": "organisation", "start": 0, "end": 2},
            {"text": "微信", "label": "product", "start": 5, "end": 7},
        ],
        "relations": [
            {"head": "騰訊", "tail": "微信", "relation": "developed"}
        ]
    })
    
    # 69. 抖音
    dataset.append({
        "text": "字節跳動開發了抖音。",
        "entities": [
            {"text": "字節跳動", "label": "organisation", "start": 0, "end": 4},
            {"text": "抖音", "label": "product", "start": 7, "end": 9},
        ],
        "relations": [
            {"head": "字節跳動", "tail": "抖音", "relation": "developed"}
        ]
    })
    
    # 70. 淘寶
    dataset.append({
        "text": "阿里巴巴開發了淘寶。",
        "entities": [
            {"text": "阿里巴巴", "label": "organisation", "start": 0, "end": 4},
            {"text": "淘寶", "label": "product", "start": 7, "end": 9},
        ],
        "relations": [
            {"head": "阿里巴巴", "tail": "淘寶", "relation": "developed"}
        ]
    })
    
    # ============================================================
    # Japanese Samples (15 samples)
    # ============================================================
    
    # 71. ソニー
    dataset.append({
        "text": "盛田昭夫はソニーの創業者です。",
        "entities": [
            {"text": "盛田昭夫", "label": "person", "start": 0, "end": 4},
            {"text": "ソニー", "label": "organisation", "start": 5, "end": 8},
        ],
        "relations": [
            {"head": "盛田昭夫", "tail": "ソニー", "relation": "founder_of"}
        ]
    })
    
    # 72. ソニー東京
    dataset.append({
        "text": "ソニーは東京に位置しています。",
        "entities": [
            {"text": "ソニー", "label": "organisation", "start": 0, "end": 3},
            {"text": "東京", "label": "location", "start": 4, "end": 6},
        ],
        "relations": [
            {"head": "ソニー", "tail": "東京", "relation": "located_in"}
        ]
    })
    
    # 73. 任天堂
    dataset.append({
        "text": "任天堂は京都に位置しています。",
        "entities": [
            {"text": "任天堂", "label": "organisation", "start": 0, "end": 3},
            {"text": "京都", "label": "location", "start": 4, "end": 6},
        ],
        "relations": [
            {"head": "任天堂", "tail": "京都", "relation": "located_in"}
        ]
    })
    
    # 74. トヨタ
    dataset.append({
        "text": "豊田喜一郎はトヨタの創業者です。",
        "entities": [
            {"text": "豊田喜一郎", "label": "person", "start": 0, "end": 5},
            {"text": "トヨタ", "label": "organisation", "start": 6, "end": 9},
        ],
        "relations": [
            {"head": "豊田喜一郎", "tail": "トヨタ", "relation": "founder_of"}
        ]
    })
    
    # 75. トヨタ CEO
    dataset.append({
        "text": "豊田章男はトヨタのCEOです。",
        "entities": [
            {"text": "豊田章男", "label": "person", "start": 0, "end": 4},
            {"text": "トヨタ", "label": "organisation", "start": 5, "end": 8},
        ],
        "relations": [
            {"head": "豊田章男", "tail": "トヨタ", "relation": "ceo_of"}
        ]
    })
    
    # 76. ソフトバンク
    dataset.append({
        "text": "孫正義はソフトバンクの創業者です。",
        "entities": [
            {"text": "孫正義", "label": "person", "start": 0, "end": 3},
            {"text": "ソフトバンク", "label": "organisation", "start": 4, "end": 10},
        ],
        "relations": [
            {"head": "孫正義", "tail": "ソフトバンク", "relation": "founder_of"}
        ]
    })
    
    # 77. ソフトバンク CEO
    dataset.append({
        "text": "孫正義はソフトバンクのCEOです。",
        "entities": [
            {"text": "孫正義", "label": "person", "start": 0, "end": 3},
            {"text": "ソフトバンク", "label": "organisation", "start": 4, "end": 10},
        ],
        "relations": [
            {"head": "孫正義", "tail": "ソフトバンク", "relation": "ceo_of"}
        ]
    })
    
    # 78. ホンダ
    dataset.append({
        "text": "本田宗一郎はホンダの創業者です。",
        "entities": [
            {"text": "本田宗一郎", "label": "person", "start": 0, "end": 5},
            {"text": "ホンダ", "label": "organisation", "start": 6, "end": 9},
        ],
        "relations": [
            {"head": "本田宗一郎", "tail": "ホンダ", "relation": "founder_of"}
        ]
    })
    
    # 79. パナソニック
    dataset.append({
        "text": "松下幸之助はパナソニックの創業者です。",
        "entities": [
            {"text": "松下幸之助", "label": "person", "start": 0, "end": 5},
            {"text": "パナソニック", "label": "organisation", "start": 6, "end": 12},
        ],
        "relations": [
            {"head": "松下幸之助", "tail": "パナソニック", "relation": "founder_of"}
        ]
    })
    
    # 80. 楽天
    dataset.append({
        "text": "三木谷浩史は楽天の創業者です。",
        "entities": [
            {"text": "三木谷浩史", "label": "person", "start": 0, "end": 5},
            {"text": "楽天", "label": "organisation", "start": 6, "end": 8},
        ],
        "relations": [
            {"head": "三木谷浩史", "tail": "楽天", "relation": "founder_of"}
        ]
    })
    
    # 81. 楽天 CEO
    dataset.append({
        "text": "三木谷浩史は楽天のCEOです。",
        "entities": [
            {"text": "三木谷浩史", "label": "person", "start": 0, "end": 5},
            {"text": "楽天", "label": "organisation", "start": 6, "end": 8},
        ],
        "relations": [
            {"head": "三木谷浩史", "tail": "楽天", "relation": "ceo_of"}
        ]
    })
    
    # 82. Nintendo Switch
    dataset.append({
        "text": "任天堂はNintendo Switchを開発しました。",
        "entities": [
            {"text": "任天堂", "label": "organisation", "start": 0, "end": 3},
            {"text": "Nintendo Switch", "label": "product", "start": 4, "end": 19},
        ],
        "relations": [
            {"head": "任天堂", "tail": "Nintendo Switch", "relation": "developed"}
        ]
    })
    
    # 83. PlayStation
    dataset.append({
        "text": "ソニーはPlayStationを開発しました。",
        "entities": [
            {"text": "ソニー", "label": "organisation", "start": 0, "end": 3},
            {"text": "PlayStation", "label": "product", "start": 4, "end": 15},
        ],
        "relations": [
            {"head": "ソニー", "tail": "PlayStation", "relation": "developed"}
        ]
    })
    
    # 84. LINE
    dataset.append({
        "text": "LINEは東京に位置しています。",
        "entities": [
            {"text": "LINE", "label": "organisation", "start": 0, "end": 4},
            {"text": "東京", "label": "location", "start": 5, "end": 7},
        ],
        "relations": [
            {"head": "LINE", "tail": "東京", "relation": "located_in"}
        ]
    })
    
    # 85. メルカリ
    dataset.append({
        "text": "山田進太郎はメルカリの創業者です。",
        "entities": [
            {"text": "山田進太郎", "label": "person", "start": 0, "end": 5},
            {"text": "メルカリ", "label": "organisation", "start": 6, "end": 10},
        ],
        "relations": [
            {"head": "山田進太郎", "tail": "メルカリ", "relation": "founder_of"}
        ]
    })
    
    # ============================================================
    # More Complex English Samples (15 samples)
    # ============================================================
    
    # 86. Complex 1
    dataset.append({
        "text": "Bill Gates and Paul Allen founded Microsoft in 1975 in Redmond.",
        "entities": [
            {"text": "Bill Gates", "label": "person", "start": 0, "end": 10},
            {"text": "Paul Allen", "label": "person", "start": 15, "end": 25},
            {"text": "Microsoft", "label": "organisation", "start": 34, "end": 43},
            {"text": "1975", "label": "date", "start": 47, "end": 51},
            {"text": "Redmond", "label": "location", "start": 55, "end": 62},
        ],
        "relations": [
            {"head": "Bill Gates", "tail": "Microsoft", "relation": "founder_of"},
            {"head": "Paul Allen", "tail": "Microsoft", "relation": "founder_of"},
            {"head": "Microsoft", "tail": "1975", "relation": "founded_in"},
            {"head": "Microsoft", "tail": "Redmond", "relation": "located_in"}
        ]
    })
    
    # 87. Complex 2
    dataset.append({
        "text": "Steve Wozniak co-founded Apple with Steve Jobs.",
        "entities": [
            {"text": "Steve Wozniak", "label": "person", "start": 0, "end": 13},
            {"text": "Apple", "label": "organisation", "start": 25, "end": 30},
            {"text": "Steve Jobs", "label": "person", "start": 36, "end": 46},
        ],
        "relations": [
            {"head": "Steve Wozniak", "tail": "Apple", "relation": "founder_of"},
            {"head": "Steve Jobs", "tail": "Apple", "relation": "founder_of"}
        ]
    })
    
    # 88. Complex 3
    dataset.append({
        "text": "Travis Kalanick founded Uber in 2009 in San Francisco.",
        "entities": [
            {"text": "Travis Kalanick", "label": "person", "start": 0, "end": 15},
            {"text": "Uber", "label": "organisation", "start": 24, "end": 28},
            {"text": "2009", "label": "date", "start": 32, "end": 36},
            {"text": "San Francisco", "label": "location", "start": 40, "end": 53},
        ],
        "relations": [
            {"head": "Travis Kalanick", "tail": "Uber", "relation": "founder_of"},
            {"head": "Uber", "tail": "2009", "relation": "founded_in"},
            {"head": "Uber", "tail": "San Francisco", "relation": "located_in"}
        ]
    })
    
    # 89. Complex 4
    dataset.append({
        "text": "Brian Chesky is the CEO of Airbnb.",
        "entities": [
            {"text": "Brian Chesky", "label": "person", "start": 0, "end": 12},
            {"text": "Airbnb", "label": "organisation", "start": 27, "end": 33},
        ],
        "relations": [
            {"head": "Brian Chesky", "tail": "Airbnb", "relation": "ceo_of"}
        ]
    })
    
    # 90. Complex 5
    dataset.append({
        "text": "Brian Chesky founded Airbnb in 2008.",
        "entities": [
            {"text": "Brian Chesky", "label": "person", "start": 0, "end": 12},
            {"text": "Airbnb", "label": "organisation", "start": 21, "end": 27},
            {"text": "2008", "label": "date", "start": 31, "end": 35},
        ],
        "relations": [
            {"head": "Brian Chesky", "tail": "Airbnb", "relation": "founder_of"},
            {"head": "Airbnb", "tail": "2008", "relation": "founded_in"}
        ]
    })
    
    # 91. Complex 6
    dataset.append({
        "text": "Patrick Collison is the CEO of Stripe.",
        "entities": [
            {"text": "Patrick Collison", "label": "person", "start": 0, "end": 16},
            {"text": "Stripe", "label": "organisation", "start": 31, "end": 37},
        ],
        "relations": [
            {"head": "Patrick Collison", "tail": "Stripe", "relation": "ceo_of"}
        ]
    })
    
    # 92. Complex 7
    dataset.append({
        "text": "Patrick Collison founded Stripe in 2010.",
        "entities": [
            {"text": "Patrick Collison", "label": "person", "start": 0, "end": 16},
            {"text": "Stripe", "label": "organisation", "start": 25, "end": 31},
            {"text": "2010", "label": "date", "start": 35, "end": 39},
        ],
        "relations": [
            {"head": "Patrick Collison", "tail": "Stripe", "relation": "founder_of"},
            {"head": "Stripe", "tail": "2010", "relation": "founded_in"}
        ]
    })
    
    # 93. Complex 8
    dataset.append({
        "text": "Daniel Ek founded Spotify in 2006 in Stockholm.",
        "entities": [
            {"text": "Daniel Ek", "label": "person", "start": 0, "end": 9},
            {"text": "Spotify", "label": "organisation", "start": 18, "end": 25},
            {"text": "2006", "label": "date", "start": 29, "end": 33},
            {"text": "Stockholm", "label": "location", "start": 37, "end": 46},
        ],
        "relations": [
            {"head": "Daniel Ek", "tail": "Spotify", "relation": "founder_of"},
            {"head": "Spotify", "tail": "2006", "relation": "founded_in"},
            {"head": "Spotify", "tail": "Stockholm", "relation": "located_in"}
        ]
    })
    
    # 94. Complex 9
    dataset.append({
        "text": "Daniel Ek is the CEO of Spotify.",
        "entities": [
            {"text": "Daniel Ek", "label": "person", "start": 0, "end": 9},
            {"text": "Spotify", "label": "organisation", "start": 24, "end": 31},
        ],
        "relations": [
            {"head": "Daniel Ek", "tail": "Spotify", "relation": "ceo_of"}
        ]
    })
    
    # 95. Complex 10
    dataset.append({
        "text": "Evan Spiegel founded Snapchat in 2011.",
        "entities": [
            {"text": "Evan Spiegel", "label": "person", "start": 0, "end": 12},
            {"text": "Snapchat", "label": "organisation", "start": 21, "end": 29},
            {"text": "2011", "label": "date", "start": 33, "end": 37},
        ],
        "relations": [
            {"head": "Evan Spiegel", "tail": "Snapchat", "relation": "founder_of"},
            {"head": "Snapchat", "tail": "2011", "relation": "founded_in"}
        ]
    })
    
    # 96. Complex 11
    dataset.append({
        "text": "Evan Spiegel is the CEO of Snapchat.",
        "entities": [
            {"text": "Evan Spiegel", "label": "person", "start": 0, "end": 12},
            {"text": "Snapchat", "label": "organisation", "start": 27, "end": 35},
        ],
        "relations": [
            {"head": "Evan Spiegel", "tail": "Snapchat", "relation": "ceo_of"}
        ]
    })
    
    # 97. Complex 12
    dataset.append({
        "text": "Drew Houston founded Dropbox in 2007 in San Francisco.",
        "entities": [
            {"text": "Drew Houston", "label": "person", "start": 0, "end": 12},
            {"text": "Dropbox", "label": "organisation", "start": 21, "end": 28},
            {"text": "2007", "label": "date", "start": 32, "end": 36},
            {"text": "San Francisco", "label": "location", "start": 40, "end": 53},
        ],
        "relations": [
            {"head": "Drew Houston", "tail": "Dropbox", "relation": "founder_of"},
            {"head": "Dropbox", "tail": "2007", "relation": "founded_in"},
            {"head": "Dropbox", "tail": "San Francisco", "relation": "located_in"}
        ]
    })
    
    # 98. Complex 13
    dataset.append({
        "text": "Stewart Butterfield founded Slack in 2013.",
        "entities": [
            {"text": "Stewart Butterfield", "label": "person", "start": 0, "end": 19},
            {"text": "Slack", "label": "organisation", "start": 28, "end": 33},
            {"text": "2013", "label": "date", "start": 37, "end": 41},
        ],
        "relations": [
            {"head": "Stewart Butterfield", "tail": "Slack", "relation": "founder_of"},
            {"head": "Slack", "tail": "2013", "relation": "founded_in"}
        ]
    })
    
    # 99. Complex 14
    dataset.append({
        "text": "Ryan Roslansky is the CEO of LinkedIn.",
        "entities": [
            {"text": "Ryan Roslansky", "label": "person", "start": 0, "end": 14},
            {"text": "LinkedIn", "label": "organisation", "start": 29, "end": 37},
        ],
        "relations": [
            {"head": "Ryan Roslansky", "tail": "LinkedIn", "relation": "ceo_of"}
        ]
    })
    
    # 100. Complex 15
    dataset.append({
        "text": "Reid Hoffman founded LinkedIn in 2002.",
        "entities": [
            {"text": "Reid Hoffman", "label": "person", "start": 0, "end": 12},
            {"text": "LinkedIn", "label": "organisation", "start": 21, "end": 29},
            {"text": "2002", "label": "date", "start": 33, "end": 37},
        ],
        "relations": [
            {"head": "Reid Hoffman", "tail": "LinkedIn", "relation": "founder_of"},
            {"head": "LinkedIn", "tail": "2002", "relation": "founded_in"}
        ]
    })
    
    return dataset


def get_dataset_statistics(dataset: List[Dict]) -> Dict:
    """Calculate dataset statistics."""
    total_samples = len(dataset)
    total_entities = sum(len(s["entities"]) for s in dataset)
    total_relations = sum(len(s.get("relations", [])) for s in dataset)
    
    # Count by relation type
    rel_counts = {}
    for sample in dataset:
        for rel in sample.get("relations", []):
            rel_type = rel["relation"]
            rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
    
    # Count by entity type
    ent_counts = {}
    for sample in dataset:
        for ent in sample["entities"]:
            ent_type = ent["label"]
            ent_counts[ent_type] = ent_counts.get(ent_type, 0) + 1
    
    return {
        "total_samples": total_samples,
        "total_entities": total_entities,
        "total_relations": total_relations,
        "entity_counts": ent_counts,
        "relation_counts": rel_counts
    }


if __name__ == "__main__":
    dataset = get_benchmark_dataset()
    stats = get_dataset_statistics(dataset)
    
    print("=" * 60)
    print("NERRE Benchmark Dataset Statistics")
    print("=" * 60)
    print(f"Total Samples:   {stats['total_samples']}")
    print(f"Total Entities:  {stats['total_entities']}")
    print(f"Total Relations: {stats['total_relations']}")
    
    print("\nEntity Type Distribution:")
    for ent_type, count in sorted(stats['entity_counts'].items()):
        print(f"  {ent_type}: {count}")
    
    print("\nRelation Type Distribution:")
    for rel_type, count in sorted(stats['relation_counts'].items()):
        print(f"  {rel_type}: {count}")
