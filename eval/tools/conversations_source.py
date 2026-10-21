"""The in-house conversations, as written: run this to regenerate eval/conversations/*.jsonl.

Each conversation has three turns. A turn is (question as a user would type it, gold
standalone rewrite, document, physical pages, verbatim evidence). The first turn is
already standalone, so its rewrite equals the question. ``switch`` marks a turn that
changes topic, where a good rewrite must drop the history rather than lean on it.
Split: within each first-turn document, sorted by id, alternate dev, test.

    python eval/tools/conversations_source.py
"""

import json
from collections import defaultdict
from pathlib import Path

NIST, QASPER, NACA = "nist-sp-800-63b-4", "qasper-2105.03011v1", "naca-rm-l56l14a"


def T(  # short, as it is written once per turn below
    question: str,
    rewrite: str | None,
    doc: str,
    pages: list[int],
    evidence: list[str],
    switch: bool = False,
) -> dict[str, object]:
    return {
        "question": question,
        "rewrite": rewrite or question,
        "doc": doc,
        "pages": pages,
        "evidence": evidence,
        "switch": switch,
    }


# One turn per line, so a diff shows exactly which turn changed.
# fmt: off
C = {
    "conv-01": [
        T("What is the minimum password length when a password is used on its own?", None, NIST, [25], ["to be a minimum of 15 characters in length"]),
        T("And if it's combined with another factor?", "What is the minimum password length when the password is used as part of multi-factor authentication?", NIST, [25], ["SHALL require them to be a minimum of eight characters in length"]),
        T("Should it be changed every year?", "Should passwords be changed periodically?", NIST, [25], ["SHALL NOT require subscribers to change passwords periodically"]),
    ],
    "conv-02": [
        T("What is the overall reauthentication timeout at AAL2?", None, NIST, [18, 22], ["SHOULD be no more than 24 hours at AAL2", "Reauthentication (recommended) 30 days overall 24 hours overall"]),
        T("What about at AAL3?", "What is the overall reauthentication timeout at AAL3?", NIST, [20, 22], ["the overall timeout for reauthentication SHALL be no more than 12 hours", "12 hours overall 15 minutes inactivity"]),
        T("And the idle limit there?", "What is the inactivity timeout at AAL3?", NIST, [20, 22], ["The inactivity timeout SHOULD be no more than 15 minutes.", "12 hours overall 15 minutes inactivity"]),
    ],
    "conv-03": [
        T("What false match rate must a biometric system meet?", None, NIST, [43], ["of one in 10000 or better for all demographic groups"]),
        T("Can voice be used?", "Can voice be used for biometric comparison?", NIST, [43], ["Biometric comparison based on voice SHALL NOT be used."]),
        T("How many failed tries does it allow?", "How many consecutive failed attempts does a biometric system allow?", NIST, [44], ["SHALL allow no more than five consecutive failed authentication"]),
    ],
    "conv-04": [
        T("How much randomness must a saved recovery code have?", None, NIST, [58], ["The recovery code SHALL include at least 64 bits from an approved"]),
        T("And an issued one?", "How many digits must an issued recovery code include?", NIST, [59], ["SHALL include at least six decimal digits (or equivalent) from an approved random bit"]),
        T("How long is it valid by email?", "How long is an issued recovery code valid when it is sent by email?", NIST, [59], ["24 hours when sent to an email address"]),
    ],
    "conv-05": [
        T("Should session cookies be readable by JavaScript?", None, NIST, [66], ["SHOULD be tagged as inaccessible via JavaScript (i.e., HttpOnly)"]),
        T("What SameSite value should they use?", "What SameSite setting should session cookies use?", NIST, [66], ["SHOULD set “SameSite=Lax” or “SameSite=Strict”."]),
        T("Separately, how far may the wireless signal for a hybrid connection reach?", "What is the maximum range of the wireless signal used for a hybrid connection?", NIST, [52], ["maximum effective range of no more than 240 meters"], switch=True),
    ],
    "conv-06": [
        T("Is out-of-band authentication phishing-resistant?", None, NIST, [29], ["Out-of-band authentication is not phishing-resistant."]),
        T("How long is its code valid?", "How long is an out-of-band authentication secret valid?", NIST, [33], ["the authentication SHALL be considered invalid unless completed within 10 minutes"]),
        T("Does asking for a new one reset the failure count?", "Does generating a new out-of-band authentication secret reset the failed authentication count?", NIST, [33], ["Generating a new authentication secret SHALL NOT reset the failed authentication count."]),
    ],
    "conv-07": [
        T("How many questions does Qasper contain?", None, QASPER, [1, 2], ["5,049 questions over 1,585"]),
        T("How were they split?", "How many Qasper questions are in the train, validation and test sets?", QASPER, [5], ["This resulted in 2,593, 1,005, and 1,451 questions"]),
        T("How much were the people who wrote them paid?", "How much were the Qasper annotators paid per hour?", QASPER, [2], ["US$25 per hour on average"]),
    ],
    "conv-08": [
        T("Which model is the Qasper baseline built on?", None, QASPER, [5], ["We use the Longformer-Encoder-Decoder (LED; Beltagy et al., 2020)"]),
        T("What GPU was it trained on?", "What GPU was the LED model trained on?", QASPER, [6], ["single RTX 8000 GPU"]),
        T("How does it do with gold evidence compared with T5?", "How does LED-base compare with T5-large when given gold evidence?", QASPER, [8], ["LED-base 54.20 24.95 44.96", "T5-large 65.59 29.11 60.03"]),
    ],
    "conv-09": [
        T("How did the TF-IDF baseline do at evidence selection?", None, QASPER, [7], ["TF-IDF 10.68 9.20"]),
        T("And the larger LED?", "How did LED-large do at evidence selection?", QASPER, [7], ["LED-large 31.25 39.37"]),
        T("What is the human lower bound for it?", "What is the human lower bound for evidence selection on Qasper?", QASPER, [5, 7], ["estimates the human performance to be 60.9 Answer-F1, and 71.6 Evidence-F1", "Human (lower bound) - 71.62"]),
    ],
    "conv-10": [
        T("In which tunnel were the flutter tests made?", None, NACA, [2, 3, 6], ["A flutter investigation has been made in the Langley transonic blowdown tunnel", "investigation was undertaken in the Langley transonic", "The tests were made in the Langley transonic blowdown tunnel"]),
        T("How big is its test section?", "How big is the test section of the Langley transonic blowdown tunnel?", NACA, [6], ["octagonal test section measuring 26 inches between flats"]),
        T("What density range can it reach?", "What density range can the Langley transonic blowdown tunnel reach?", NACA, [6], ["from approximately 0.001 to 0.012 slug per cubic foot"]),
    ],
    "conv-11": [
        T("What were the flutter models made of?", None, NACA, [5], ["a phenolic laminate material with high-strength paper reinforcement"]),
        T("Who wrote the report?", "Who wrote the NACA report on the flutter of the 10-degree sweptback wing?", NACA, [1, 2], ["By George W. Jones, Jr.", "ByGeorgeW. fones, Jr."]),
        T("Different topic: in Qasper, how long are extractive answers?", "How long are extractive answers in Qasper on average?", QASPER, [3], ["The average length of the extractive answers is 14.4 words"], switch=True),
    ],
    "conv-12": [
        T("What is the minimum length of a look-up secret?", None, NIST, [28], ["Look-up secrets SHALL be at least six decimal digits (or equivalent) in length."]),
        T("How many times can each be used?", "How many times can a look-up secret be used successfully?", NIST, [28], ["SHALL be used successfully only once"]),
        T("Now about the Qasper paper: which tool was used for the annotators' qualification exams?", "Which tool was used to build qualification exams for the Qasper annotators?", QASPER, [3], ["using CrowdAQ (Ning et al., 2020)"], switch=True),
    ],
}
# fmt: on


def split() -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for cid, turns in C.items():
        groups[str(turns[0]["doc"])].append(cid)
    out: dict[str, list[dict[str, object]]] = {"dev": [], "test": []}
    for doc in sorted(groups):
        for i, cid in enumerate(sorted(groups[doc])):
            out["dev" if i % 2 == 0 else "test"].append({"id": cid, "lang": "en", "turns": C[cid]})
    return {name: sorted(rows, key=lambda r: str(r["id"])) for name, rows in out.items()}


def main() -> None:
    folder = Path("eval/conversations")
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in split().items():
        lines = [json.dumps(r, ensure_ascii=False) for r in rows]
        (folder / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(name, len(rows))


if __name__ == "__main__":
    main()
