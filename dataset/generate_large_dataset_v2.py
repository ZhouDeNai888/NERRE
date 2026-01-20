#!/usr/bin/env python3
"""
Generate a large multilingual training dataset for NERRE.
Version 2: 100 Entity Types + 100 Relation Types
Target: 10,000+ samples across 5 languages (English, Chinese, Japanese, Korean, Thai)
"""

import json
import random
from typing import List, Dict, Any, Tuple

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
    "organisation": "A general group of people, association, or organized body with a particular purpose, not fitting other specific categories.",
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


}

# ============================================================================
# 100 RELATION TYPES - Organized by Category
# ============================================================================

RELATION_TYPES = {
    # === CREATION/OWNERSHIP (15 types) ===
    "founder_of": ("person", "organisation", "Founded/created an organization"),
    "ceo_of": ("person", "organisation", "Is CEO/leader of"),
    "owner_of": ("person", "organisation", "Owns"),
    "creator_of": ("person", "product", "Created a product/work"),
    "inventor_of": ("person", "invention", "Invented"),
    "author_of": ("person", "book", "Wrote/authored"),
    "director_of": ("person", "movie", "Directed a film"),
    "producer_of": ("person", "product", "Produced"),
    "designed_by": ("product", "person", "Was designed by"),
    "developed": ("organisation", "product", "Developed a product"),
    "manufactured_by": ("product", "organisation", "Made by"),
    "published_by": ("book", "organisation", "Published by"),
    "composed_by": ("music_album", "musician", "Composed/written by"),
    "painted_by": ("artwork", "artist", "Painted by"),
    "patented_by": ("patent", "person", "Patented by"),
    
    # === LOCATION (12 types) ===
    "located_in": ("organisation", "location", "Located in"),
    "headquartered_in": ("company", "city", "Has headquarters in"),
    "born_in": ("person", "location", "Was born in"),
    "died_in": ("person", "location", "Died in"),
    "lives_in": ("person", "city", "Lives/resides in"),
    "operates_in": ("company", "country", "Operates in"),
    "based_in": ("organisation", "city", "Based in"),
    "filmed_in": ("movie", "location", "Filmed in"),
    "held_in": ("event", "location", "Held/took place in"),
    "native_to": ("person", "country", "Native to"),
    "capital_of": ("city", "country", "Is capital of"),
    "part_of": ("location", "location", "Is part of"),
    
    # === TIME (10 types) ===
    "founded_in": ("organisation", "date", "Founded in year"),
    "released_in": ("product", "date", "Released in year"),
    "born_on": ("person", "date", "Born on date"),
    "died_on": ("person", "date", "Died on date"),
    "started_in": ("event", "date", "Started in"),
    "ended_in": ("event", "date", "Ended in"),
    "established_in": ("organisation", "date", "Established in"),
    "occurred_on": ("event", "date", "Occurred on"),
    "graduated_in": ("person", "date", "Graduated in"),
    "married_on": ("person", "date", "Married on"),
    
    # === EMPLOYMENT (10 types) ===
    "works_at": ("person", "organisation", "Works at"),
    "employed_by": ("person", "company", "Employed by"),
    "position_at": ("person", "organisation", "Has position at"),
    "manages": ("person", "organisation", "Manages"),
    "leads": ("person", "organisation", "Leads"),
    "reports_to": ("person", "person", "Reports to"),
    "hired_by": ("person", "company", "Was hired by"),
    "resigned_from": ("person", "company", "Resigned from"),
    "retired_from": ("person", "organisation", "Retired from"),
    "consultant_for": ("person", "company", "Consultant for"),
    
    # === EDUCATION (8 types) ===
    "studied_at": ("person", "university", "Studied at"),
    "graduated_from": ("person", "university", "Graduated from"),
    "degree_from": ("person", "university", "Got degree from"),
    "professor_at": ("scientist", "university", "Professor at"),
    "teaches_at": ("person", "school", "Teaches at"),
    "research_at": ("scientist", "research_institute", "Does research at"),
    "alumni_of": ("person", "university", "Alumni of"),
    "dropout_from": ("person", "university", "Dropped out from"),
    
    # === FAMILY (8 types) ===
    "spouse_of": ("person", "person", "Married to"),
    "parent_of": ("person", "person", "Parent of"),
    "child_of": ("person", "person", "Child of"),
    "sibling_of": ("person", "person", "Sibling of"),
    "relative_of": ("person", "person", "Relative of"),
    "married_to": ("person", "person", "Married to"),
    "divorced_from": ("person", "person", "Divorced from"),
    "partner_of": ("person", "person", "Partner of"),
    
    # === BUSINESS (12 types) ===
    "subsidiary_of": ("company", "company", "Subsidiary of"),
    "acquired_by": ("company", "company", "Acquired by"),
    "merged_with": ("company", "company", "Merged with"),
    "partner_with": ("company", "company", "Partner with"),
    "competitor_of": ("company", "company", "Competes with"),
    "investor_in": ("person", "company", "Invested in"),
    "invested_by": ("startup", "company", "Received investment from"),
    "supplies_to": ("company", "company", "Supplies to"),
    "customer_of": ("company", "company", "Customer of"),
    "distributor_of": ("company", "product", "Distributes"),
    "licensed_by": ("product", "company", "Licensed by"),
    "sponsored_by": ("event", "company", "Sponsored by"),
    
    # === ASSOCIATION (8 types) ===
    "member_of": ("person", "organisation", "Member of"),
    "affiliated_with": ("person", "organisation", "Affiliated with"),
    "belongs_to": ("product", "company", "Belongs to"),
    "represents": ("person", "country", "Represents"),
    "ambassador_for": ("person", "organisation", "Ambassador for"),
    "spokesperson_for": ("person", "company", "Spokesperson for"),
    "endorses": ("person", "product", "Endorses"),
    "supports": ("person", "political_party", "Supports"),
    
    # === AWARDS & ACHIEVEMENTS (7 types) ===
    "won": ("person", "award", "Won award"),
    "nominated_for": ("person", "award", "Nominated for"),
    "recipient_of": ("person", "award", "Recipient of"),
    "awarded_by": ("award", "organisation", "Awarded by"),
    "achieved": ("person", "title", "Achieved title"),
    "holds_record": ("person", "event", "Holds record in"),
    "champion_of": ("athlete", "competition", "Champion of"),
    
    # === MEDIA & ENTERTAINMENT (10 types) ===
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


    # === FINANCIAL (8 types) ===
    "has_net_worth": ("person", "money", "Has a total estimated net worth of"),
    "valuation_of": ("money", "company", "The market valuation of a company"),
    "invested_amount": ("person", "money", "Amount invested by a person"),
    "holds_shares_of": ("person", "percent", "Percentage of shares held in a company"),
    "market_cap": ("company", "money", "Market capitalization of a company"),
    "sold_for": ("product", "money", "Product or company sold for this amount"),
    "revenue_of": ("company", "money", "Annual or period revenue of a company"),
    "salary_of": ("person", "money", "Estimated salary or compensation of a person"),


    # === DIGITAL & MARKET (5 types) ===
    "official_website": ("organisation", "url", "The official URL of an organization"),
    "listed_as": ("company", "stock_symbol", "Company is listed under this ticker symbol"),
    "download_url": ("software", "url", "The download link for a software/app"),
    "social_media": ("person", "url", "Social media profile link of a person"),
    "trading_on": ("stock_symbol", "bank", "Stock symbol traded on a specific exchange"),


    # === HEALTHCARE (5 types) ===
    "treats": ("medicine", "disease", "Medicine or drug used to treat a condition"),
    "diagnosed_with": ("person", "disease", "Person diagnosed with a medical condition"),
    "affects": ("disease", "organ", "Disease that affects specific body parts"),
    "dosage_of": ("medicine", "quantity", "Recommended dosage amount"),
    "developed_vaccine": ("company", "medicine", "Company developed a specific vaccine"),


    "governs": ("organisation", "location", "Governs/rules over a territory"),
    "head_of_state": ("person", "country", "Is the primary leader of a country"),
    "member_of_parliament": ("person", "organisation", "Member of a legislative body"),
    "allied_with": ("country", "country", "Has a formal alliance with"),
    "sanctioned_by": ("person", "organisation", "Was sanctioned or penalized by"),
    "ratified": ("organisation", "legal_document", "Formally approved a treaty or law"),
    "vetoed_by": ("legal_document", "person", "Was rejected by a leader"),
    "enforced_by": ("legal_document", "organisation", "Is implemented by an agency"),

    "scientific_discovery": ("scientist", "invention", "Discovered a new phenomenon/entity"),
    "published_in": ("research_paper", "journal", "Was published in a specific journal"),
    "cited_by": ("research_paper", "research_paper", "Was referenced by another study"),
    "collaborated_with": ("scientist", "scientist", "Worked together on research"),
    "funded_by": ("research_at", "organisation", "Research was financed by"),
    "hypothesis_of": ("theory", "scientist", "Proposed as a scientific explanation"),
    "clinical_trial_of": ("medicine", "disease", "Undergoing testing for a condition"),
    "sequenced": ("scientist", "gene", "Mapped the genetic sequence of"),
    "peer_reviewed_by": ("research_paper", "scientist", "Verified by an independent expert"),
    "experimental_data_from": ("research_paper", "research_institute", "Data collected at this facility"),


    "launched_by": ("satellite", "organisation", "Sent into space by"),
    "orbits": ("celestial_object", "celestial_object", "Moves in orbit around"),
    "landed_on": ("spacecraft", "celestial_object", "Successfully landed on"),
    "observed_by": ("celestial_object", "research_institute", "Monitored by an observatory"),
    "mission_of": ("spacecraft", "organisation", "Is a project of a space agency"),
    "reusable_launch_vehicle": ("rocket", "organisation", "Developed a reusable rocket"),


    "plaintiff_in": ("person", "legal_case", "The party bringing the lawsuit"),
    "defendant_in": ("person", "legal_case", "The party being sued/accused"),
    "presided_over_by": ("legal_case", "judge", "The judge in charge of the case"),
    "convicted_of": ("person", "crime", "Found guilty of a specific offense"),
    "settled_with": ("company", "company", "Reached an out-of-court agreement"),
    "infringes_on": ("product", "patent", "Violates existing intellectual property"),
    "compliant_with": ("organisation", "legal_document", "Follows specific regulations"),

    "exhibited_at": ("artwork", "museum", "Currently displayed at"),
    "discovered_at": ("archaeological_site", "location", "Site was found in"),
    "excavated_by": ("archaeological_site", "scientist", "Systematically dug up by"),
    "dated_to": ("artifact", "date", "Origins traced back to this era"),
    "restored_by": ("artwork", "organisation", "Repaired/preserved by"),
    "historical_figure_in": ("person", "event", "A key person in a past event"),
    "influenced_by": ("artist", "person", "Artistic style shaped by"),
    "dedicated_to": ("monument", "person", "Built in honor of"),






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
        "Justin Trudeau", "Boris Johnson", "Vladimir Putin", "Xi Jinping", "Narendra Modi"
    ],
    "scientist": [
        "Albert Einstein", "Stephen Hawking", "Marie Curie", "Isaac Newton", "Charles Darwin",
        "Nikola Tesla", "Richard Feynman", "Neil deGrasse Tyson", "Michio Kaku", "Jane Goodall",
        "Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio", "Fei-Fei Li", "Demis Hassabis"
    ],
    "artist": [
        "Leonardo da Vinci", "Pablo Picasso", "Vincent van Gogh", "Claude Monet", "Andy Warhol",
        "Salvador Dalí", "Frida Kahlo", "Banksy", "Yayoi Kusama", "Ai Weiwei"
    ],
    "athlete": [
        "Michael Jordan", "LeBron James", "Cristiano Ronaldo", "Lionel Messi", "Serena Williams",
        "Tiger Woods", "Roger Federer", "Usain Bolt", "Muhammad Ali", "Tom Brady",
        "Naomi Osaka", "Lewis Hamilton", "Michael Phelps", "Simone Biles", "Kobe Bryant"
    ],
    "musician": [
        "Taylor Swift", "Beyoncé", "Ed Sheeran", "Drake", "The Weeknd",
        "BTS", "Ariana Grande", "Bruno Mars", "Lady Gaga", "Rihanna",
        "Adele", "Justin Bieber", "Billie Eilish", "Coldplay", "Dua Lipa"
    ],
    "actor": [
        "Leonardo DiCaprio", "Tom Hanks", "Meryl Streep", "Robert Downey Jr.", "Scarlett Johansson",
        "Dwayne Johnson", "Jennifer Lawrence", "Brad Pitt", "Angelina Jolie", "Chris Hemsworth",
        "Keanu Reeves", "Will Smith", "Emma Watson", "Timothée Chalamet", "Zendaya"
    ],
    "director": [
        "Steven Spielberg", "Christopher Nolan", "Martin Scorsese", "Quentin Tarantino", "James Cameron",
        "Denis Villeneuve", "Greta Gerwig", "Bong Joon-ho", "Ridley Scott", "Peter Jackson"
    ],
    "author": [
        "J.K. Rowling", "Stephen King", "George R.R. Martin", "Dan Brown", "Haruki Murakami",
        "Margaret Atwood", "Neil Gaiman", "Yuval Noah Harari", "Malcolm Gladwell", "James Clear"
    ],
    "entrepreneur": [
        "Richard Branson", "Jack Ma", "Larry Ellison", "Michael Bloomberg", "Oprah Winfrey",
        "Marc Benioff", "Reid Hoffman", "Peter Thiel", "Travis Kalanick", "Brian Chesky"
    ],
    "engineer": [
        "Linus Torvalds", "Guido van Rossum", "Brendan Eich", "James Gosling", "Dennis Ritchie",
        "Ken Thompson", "Bjarne Stroustrup", "Anders Hejlsberg", "John Carmack", "Margaret Hamilton"
    ],
    "doctor": [
        "Anthony Fauci", "Sanjay Gupta", "Oz Mehmet", "Ben Carson", "Atul Gawande"
    ],
    "journalist": [
        "Anderson Cooper", "Christiane Amanpour", "Wolf Blitzer", "Rachel Maddow", "Tucker Carlson"
    ],
    "chef": [
        "Gordon Ramsay", "Jamie Oliver", "Anthony Bourdain", "Wolfgang Puck", "Massimo Bottura"
    ],
    
    # Organizations
    "organisation": [
        "United Nations", "World Health Organization", "Red Cross", "Amnesty International", "Greenpeace"
    ],
    "company": [
        "Apple", "Microsoft", "Google", "Amazon", "Meta", "Tesla", "NVIDIA", "Intel", "AMD", "IBM",
        "Oracle", "Salesforce", "Adobe", "Netflix", "Spotify", "Uber", "Airbnb", "Twitter", "LinkedIn"
    ],
    "startup": [
        "OpenAI", "Anthropic", "Stripe", "Databricks", "Canva", "Figma", "Notion", "Airtable", "Vercel"
    ],
    "university": [
        "Harvard University", "Stanford University", "MIT", "Oxford University", "Cambridge University",
        "Yale University", "Princeton University", "Columbia University", "UC Berkeley", "Caltech"
    ],
    "sports_team": [
        "Los Angeles Lakers", "New York Yankees", "Real Madrid", "Barcelona FC", "Manchester United",
        "Golden State Warriors", "Dallas Cowboys", "New England Patriots", "Chicago Bulls"
    ],
    "bank": [
        "JPMorgan Chase", "Bank of America", "Goldman Sachs", "Morgan Stanley", "Citibank",
        "Wells Fargo", "HSBC", "Deutsche Bank", "Credit Suisse", "Barclays"
    ],
    "airline": [
        "United Airlines", "Delta Airlines", "American Airlines", "Emirates", "Singapore Airlines",
        "Lufthansa", "British Airways", "Qatar Airways", "Air France", "Southwest Airlines"
    ],
    "media_company": [
        "CNN", "BBC", "The New York Times", "The Washington Post", "Reuters",
        "Bloomberg", "Fox News", "MSNBC", "The Guardian", "Wall Street Journal"
    ],
    "research_institute": [
        "NASA", "CERN", "NIH", "Max Planck Institute", "MIT Media Lab",
        "DeepMind", "OpenAI Research", "Google Brain", "FAIR", "Microsoft Research"
    ],
    "hospital": [
        "Mayo Clinic", "Cleveland Clinic", "Johns Hopkins Hospital", "Massachusetts General Hospital"
    ],
    "manufacturer": [
        "Samsung Electronics", "Foxconn", "TSMC", "Qualcomm", "Broadcom", "Texas Instruments"
    ],
    "retailer": [
        "Walmart", "Amazon", "Costco", "Target", "Home Depot", "Best Buy", "IKEA"
    ],
    
    # Locations
    "location": [
        "Silicon Valley", "Wall Street", "Hollywood", "Times Square", "Central Park"
    ],
    "city": [
        "San Francisco", "New York", "Los Angeles", "Seattle", "Boston", "Chicago", "Austin",
        "London", "Paris", "Tokyo", "Singapore", "Hong Kong", "Shanghai", "Beijing", "Seoul",
        "Sydney", "Toronto", "Berlin", "Amsterdam", "Dubai", "Mumbai", "Bangalore"
    ],
    "country": [
        "United States", "China", "Japan", "Germany", "United Kingdom", "France", "India",
        "Canada", "Australia", "South Korea", "Brazil", "Italy", "Spain", "Russia"
    ],
    "state": [
        "California", "Texas", "New York", "Florida", "Washington", "Massachusetts", "Colorado"
    ],
    "building": [
        "Empire State Building", "Burj Khalifa", "One World Trade Center", "Taipei 101"
    ],
    "landmark": [
        "Eiffel Tower", "Statue of Liberty", "Great Wall of China", "Taj Mahal", "Colosseum"
    ],
    "stadium": [
        "Madison Square Garden", "Wembley Stadium", "Camp Nou", "Yankee Stadium"
    ],
    
    # Products & Technology
    "product": [
        "iPhone", "MacBook", "iPad", "Apple Watch", "AirPods", "Tesla Model S", "PlayStation 5"
    ],
    "software": [
        "Windows", "macOS", "Microsoft Office", "Adobe Photoshop", "Slack", "Zoom", "Notion"
    ],
    "app": [
        "Instagram", "TikTok", "WhatsApp", "Snapchat", "Uber", "Spotify", "Netflix"
    ],
    "game": [
        "Minecraft", "Fortnite", "League of Legends", "Call of Duty", "Grand Theft Auto V",
        "The Legend of Zelda", "Super Mario", "Pokemon", "FIFA", "Elden Ring"
    ],
    "movie": [
        "Avatar", "Titanic", "Avengers: Endgame", "The Dark Knight", "Inception",
        "Interstellar", "The Matrix", "Star Wars", "Jurassic Park", "The Godfather"
    ],
    "book": [
        "Harry Potter", "The Lord of the Rings", "A Song of Ice and Fire", "The Da Vinci Code",
        "Sapiens", "Atomic Habits", "The Lean Startup", "Zero to One"
    ],
    "music_album": [
        "Thriller", "The Dark Side of the Moon", "Abbey Road", "Back in Black",
        "1989", "25", "Divide", "Scorpion"
    ],
    "programlang": [
        "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust", "Swift", "Kotlin"
    ],
    "framework": [
        "React", "Angular", "Vue.js", "Django", "Flask", "Spring", "TensorFlow", "PyTorch"
    ],
    "database": [
        "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "Oracle Database"
    ],
    "ai_model": [
        "GPT-4", "ChatGPT", "Claude", "Gemini", "LLaMA", "DALL-E", "Midjourney", "Stable Diffusion"
    ],
    "os": [
        "Windows 11", "macOS Sonoma", "Linux", "Ubuntu", "Android", "iOS"
    ],
    "cryptocurrency": [
        "Bitcoin", "Ethereum", "Solana", "Cardano", "Dogecoin", "XRP"
    ],
    
    # Events & Awards
    "event": [
        "CES", "WWDC", "Google I/O", "AWS re:Invent", "Mobile World Congress"
    ],
    "conference": [
        "TED", "Davos Forum", "NeurIPS", "ICML", "CVPR", "ACL"
    ],
    "competition": [
        "Olympics", "World Cup", "Super Bowl", "Wimbledon", "Tour de France"
    ],
    "award": [
        "Nobel Prize", "Academy Award", "Grammy Award", "Emmy Award", "Pulitzer Prize",
        "Turing Award", "Fields Medal", "Golden Globe", "BAFTA"
    ],
    "tv_show": [
        "Game of Thrones", "Breaking Bad", "Stranger Things", "The Office", "Friends"
    ],
    
    # Time
    "date": ["2024", "2023", "2022", "2021", "2020", "2019", "2010", "2000", "1990", "1980"],
    "year": ["2024", "2023", "2022", "2021", "2020"],
    "month": ["January", "February", "March", "April", "May", "June"],
    "century": ["21st century", "20th century", "19th century"],


    # === FINANCIAL & NUMERIC ===
    "money": [
        "1 billion dollars", "$44 billion", "100 million euros", "£500,000", 
        "10.5 billion USD", "50 million THB", "net worth of $200B"
    ],
    "percent": [
        "15%", "51 percent", "0.5%", "99.9%", "a quarter", "ten percent"
    ],
    "stock_symbol": [
        "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "NFLX"
    ],

    # === DIGITAL & INFRASTRUCTURE ===
    "url": [
        "https://www.openai.com", "www.google.com", "github.com/trending", 
        "apple.co/support", "https://t.co/xyz123"
    ],
    "email": [
        "contact@tesla.com", "support@apple.com", "ceo@microsoft.com", 
        "info@un.org", "admin@stanford.edu"
    ],
    "phone_number": [
        "+1-555-0199", "02-123-4567", "+44 20 7946 0958", "1-800-APPLE"
    ],

    # === LEGAL & MEDICAL ===
    "legal_document": [
        "GDPR", "Section 301", "Article 50", "The Constitution", 
        "Patent Act", "Digital Millennium Copyright Act"
    ],
    "disease": [
        "COVID-19", "Diabetes", "Alzheimer's", "Influenza", "Hypertension", "Cancer"
    ],
    "medicine": [
        "Paracetamol", "Insulin", "Pfizer vaccine", "Aspirin", "Ibuprofen"
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
        "蔡英文", "習近平", "李克強", "王毅", "賴清德"
    ],
    "scientist": [
        "屠呦呦", "楊振寧", "李政道", "丁肇中", "錢學森", "袁隆平"
    ],
    "athlete": [
        "姚明", "劉翔", "李娜", "蘇炳添", "谷愛凌", "朱婷"
    ],
    "musician": [
        "周杰倫", "林俊傑", "蔡依林", "張惠妹", "五月天", "鄧紫棋"
    ],
    "actor": [
        "成龍", "李連杰", "周潤發", "梁朝偉", "劉德華", "章子怡", "鞏俐"
    ],
    "director": [
        "張藝謀", "李安", "王家衛", "陳凱歌", "馮小剛"
    ],
    "author": [
        "莫言", "余華", "劉慈欣", "金庸", "瓊瑤"
    ],
    "company": [
        "台積電", "鴻海", "聯發科", "華碩", "宏碁", "阿里巴巴", "騰訊", "百度", "華為", "小米",
        "京東", "美團", "字節跳動", "拼多多", "滴滴", "網易", "聯想", "比亞迪", "寧德時代"
    ],
    "startup": [
        "商湯科技", "曠視科技", "依圖科技", "雲從科技", "字節跳動"
    ],
    "university": [
        "北京大學", "清華大學", "復旦大學", "上海交通大學", "浙江大學",
        "台灣大學", "成功大學", "交通大學", "中央大學", "中山大學"
    ],
    "sports_team": [
        "中國國家足球隊", "中華台北隊", "廣州恆大", "北京國安", "上海上港"
    ],
    "bank": [
        "中國工商銀行", "中國建設銀行", "中國銀行", "國泰世華銀行", "中國信託"
    ],
    "city": [
        "台北", "新竹", "台中", "高雄", "北京", "上海", "深圳", "杭州", "廣州", "成都",
        "香港", "澳門", "南京", "武漢", "西安", "青島", "廈門"
    ],
    "country": [
        "中國", "台灣", "日本", "韓國", "新加坡", "美國"
    ],
    "product": [
        "微信", "QQ", "抖音", "TikTok", "支付寶", "淘寶", "天貓", "京東商城",
        "華為Mate", "小米手機", "OPPO", "vivo", "比亞迪電動車"
    ],
    "movie": [
        "戰狼2", "流浪地球", "哪吒之魔童降世", "長津湖", "紅海行動"
    ],
    "book": [
        "三體", "活著", "圍城", "紅樓夢", "射雕英雄傳"
    ],
    "award": [
        "金馬獎", "金鐘獎", "金曲獎", "中國電影金雞獎", "華語電影傳媒大獎"
    ],
    "date": ["2024年", "2023年", "2022年", "2021年", "2020年", "2010年", "2000年", "1990年"],
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
        "nonprofit": "organisation",
        "government_agency": "organisation",
        "school": "university",
        "hospital": "organisation",
        "military": "organisation",
        "political_party": "organisation",
        "research_institute": "university",
        "museum": "organisation",
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

     ("The official website of {organisation} is {url}.", 
     [("organisation", "organisation"), ("url", "url")], 
     [("organisation", "url", "official_website")]),
    
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

     ("เว็บไซต์ทางการของ {organisation} คือ {url}", 
     [("organisation", "organisation"), ("url", "url")], 
     [("organisation", "url", "official_website")]),

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
        "en": {"person": "He", "organisation": "It", "company": "It", "default": "It", "suffix": " is leading the field."},
        "th": {"person": "เขา", "organisation": "มัน", "company": "องค์กรนี้", "default": "สิ่งนี้", "suffix": "กำลังเป็นผู้นำในอุตสาหกรรม"},
        "zh": {"person": "他", "organisation": "它", "company": "該公司", "default": "它", "suffix": "目前在行業中處於領先地位"},
        "ja": {"person": "彼", "organisation": "それ", "company": "同社", "default": "それ", "suffix": "は現在業界をリードしています"},
        "ko": {"person": "그", "organisation": "그것", "company": "이 회사는", "default": "그것", "suffix": " 현재 업계를 선도하고 있습니다"}
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
        
        entities.append({
            "start": pronoun_start,
            "end": pronoun_start + len(pronoun),
            "label": label, # ใช้ Label เดียวกับตัวหลัก
            "text": pronoun
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
            entities.append({
                "start": chosen_noise.find(fake_val),
                "end": chosen_noise.find(fake_val) + len(fake_val),
                "label": fake_type,
                "text": fake_val
            })
        else:
            # กรณี Noise อยู่ข้างหลัง (ไม่ต้องแก้ Offset)
            start_pos = len(text) + chosen_noise.find(fake_val)
            text = text + chosen_noise
            entities.append({
                "start": start_pos,
                "end": start_pos + len(fake_val),
                "label": fake_type,
                "text": fake_val
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
    """สลับโครงสร้างประโยค (Semantic Reversal) เช่น Active เป็น Passive"""
    # ตัวอย่างเฉพาะทางสำหรับ Relation บางประเภท
    reversal_map = {
        "founder_of": "{tail} was founded by {head}",
        "ceo_of": "The CEO of {tail} is {head}",
        "developed": "{tail} was developed by {head}"
    }
    
    new_template = template
    new_rels = relation_defs
    
    if random.random() < 0.3: # 30% สลับโครงสร้าง
        for rel_type, new_fmt in reversal_map.items():
            for i, (h, t, r) in enumerate(relation_defs):
                if r == rel_type:
                    # สร้าง Template ใหม่โดยอ้างอิง Placeholder เดิม
                    new_template = new_fmt.format(head="{" + h + "}", tail="{" + t + "}")
                    break
    
    return new_template, entity_defs, new_rels

def get_interleaving_noise(entities_dict: Dict) -> str:
    """สร้าง Entity ขวาง (Interleaving) ที่ไม่เกี่ยวข้องกัน"""
    fake_type = random.choice(["person", "company", "city"])
    fake_val = get_entity(entities_dict, fake_type)
    distractors = [
        f", along with {fake_val},",
        f" (while {fake_val} was absent)",
        f" and {fake_val}"
    ]
    return random.choice(distractors)


# -----------------------------------------------------------------
#  generate_sample function
# -----------------------------------------------------------------

def generate_sample(templates: List, entities_dict: Dict) -> Dict:
    template_raw, entity_defs, relation_defs = random.choice(templates)


    # [แทรกจุดที่ 1] สลับโครงสร้าง Active/Passive ก่อนเริ่มกระบวนการอื่น
    template_raw, entity_defs, relation_defs = apply_semantic_reversal(template_raw, entity_defs, relation_defs)
    
    # 1. Synonym Replacement ในตัว Template เอง
    template = apply_synonym(template_raw)
    
    replacements = {}
    used_values = set()
    
    # 2. เตรียมข้อมูลสำหรับแต่ละ Placeholder
    for placeholder, entity_type in entity_defs:
        key = "{" + placeholder + "}"
        original_value = get_entity(entities_dict, entity_type)
        
        # ป้องกันค่าซ้ำ
        while original_value in used_values:
            original_value = get_entity(entities_dict, entity_type)
        
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
        noise = get_interleaving_noise(entities_dict)
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
    if random.random() < 0.3: # 30% ของข้อมูลจะมีตัวหลอก
        random_type = random.choice(list(entities_dict.keys()))
        fake_entity = get_entity(entities_dict, random_type)
        if fake_entity not in used_values:
            filler_phrases = [
                f" (similar to {fake_entity})",
                f" unlike {fake_entity}",
                f". Note: {fake_entity} was not involved."
            ]
            text += random.choice(filler_phrases)


    # [แทรกจุดที่ 4] ตัดสินใจว่าจะเป็นเคส "ไม่มีความสัมพันธ์" หรือไม่
    is_negative_case = random.random() < 0.15

    # 5. Build entities metadata
    entities = []
    for placeholder, entity_type in entity_defs:
        key = "{" + placeholder + "}"
        entity_text = replacements[key]
        
        # ค้นหาตำแหน่งที่ถูกต้อง (ระวังคำซ้ำ)
        start = text.find(entity_text)
        if start != -1:
            entities.append({
                "start": start,
                "end": start + len(entity_text),
                "label": entity_type,
                "text": entity_text
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
                    relations.append({
                        "head": head_text,
                        "tail": tail_text,
                        "label": rel_type
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


def generate_samples(templates: List, entities_dict: Dict, count: int) -> List[Dict]:
    """Generate multiple samples."""
    samples = []
    for _ in range(count):
        sample = generate_sample(templates, entities_dict)
        if sample["entities"] and sample["relations"]:  # Only add valid samples
            samples.append(sample)
    return samples


def generate_dataset(target_count: int = 10000) -> List[Dict]:
    """Generate a balanced multilingual dataset."""
    
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


if __name__ == "__main__":
    
    # if True:
    #     print_type_summary()
    #     exit(0)
    
    random.seed(42)
    
    print(f"NERRE Dataset Generator v2")
    print(f"Entity Types: {len(ENTITY_TYPES)}")
    print(f"Relation Types: {len(RELATION_TYPES)}")
    print(f"Generating 100000 multilingual samples...")
    
    samples = generate_dataset(100000)
    
    print_statistics(samples)
    
    output_path = f"/data/tcustpg18/NERRE/NERRE/dataset/multilingual_data_v7_100000.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"\nDataset saved to: {output_path}")
