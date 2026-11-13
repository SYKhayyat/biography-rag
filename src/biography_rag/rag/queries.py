"""The 20 fixed questions. ``keywords`` are generic retrieval hints for the keyword arm of
hybrid search: words a biography tends to use for that topic, never facts about any one
person (so they cannot smuggle knowledge into the answer)."""

from __future__ import annotations

from biography_rag.domain import QueryTemplate

QUERIES: tuple[QueryTemplate, ...] = (
    QueryTemplate(1, "Where and when was {person} born?", "born birth birthplace birthday native town year"),
    QueryTemplate(
        2,
        "Where were {person}'s parents from, and what did they do?",
        "father mother parents trade occupation emigrated came from native",
    ),
    QueryTemplate(3, "What was {person}'s childhood like?", "childhood boyhood girlhood boy girl young early years"),
    QueryTemplate(
        4, "Who were {person}'s siblings?", "brother brothers sister sisters siblings eldest youngest family"
    ),
    QueryTemplate(
        5,
        "What schooling or training did {person} have?",
        "school schooling education taught learned studied apprentice apprenticed",
    ),
    QueryTemplate(6, "What was {person}'s first job?", "first job work employed apprentice trade worked shop"),
    QueryTemplate(
        7, "An adventurous story from {person}'s life.", "adventure journey voyage expedition ran away escape"
    ),
    QueryTemplate(
        8,
        "A scary or dangerous moment {person} faced.",
        "danger dangerous peril narrowly escaped storm shipwreck fear threatened",
    ),
    QueryTemplate(9, "A funny anecdote about {person}.", "laughed laughter humor humour joke amusing wit comical"),
    QueryTemplate(
        10,
        "A time {person} failed or made a serious mistake.",
        "mistake failure failed error erratum regretted blunder lost",
    ),
    QueryTemplate(
        11,
        "The hardest thing {person} went through.",
        "grief loss sorrow suffering illness hardship misfortune death",
    ),
    QueryTemplate(
        12,
        "Who did {person} marry, and what was that relationship like?",
        "married marriage wife husband wedding courtship",
    ),
    QueryTemplate(13, "What about {person}'s children?", "son daughter children child born died"),
    QueryTemplate(
        14,
        "Who were {person}'s closest friends or mentors?",
        "friend friends friendship mentor patron companion intimate",
    ),
    QueryTemplate(
        15, "Who were {person}'s rivals or enemies?", "enemy enemies rival quarrel opponent hostile dispute attacked"
    ),
    QueryTemplate(
        16, "What is {person} best known for?", "famous fame known celebrated renowned achievement reputation"
    ),
    QueryTemplate(
        17,
        "Something {person} made, invented or discovered.",
        "invented invention discovered discovery experiment designed contrivance",
    ),
    QueryTemplate(
        18,
        "Where did {person} travel or live?",
        "travelled traveled voyage abroad lived residence moved resided city",
    ),
    QueryTemplate(
        19, "A habit, hobby or quirk of {person}.", "habit habits custom hobby fond amusement practice daily"
    ),
    QueryTemplate(
        20,
        "How did {person} die, and how is {person} remembered?",
        "died death funeral buried grave memory legacy honored remembered",
    ),
)
