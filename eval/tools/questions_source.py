"""The in-house questions, as written: run this to regenerate eval/questions/*.jsonl.

Each entry: (id, question, doc, physical pages, evidence quotes, reference answer, kind).
An empty page list marks an unanswerable question. Evidence quotes are verbatim from the
cited pages (a test checks every one). The split into dev and test is deterministic:
within each (document, kind, answerable) group, sorted by id, alternate dev, test.

    python eval/tools/questions_source.py
"""

import json
from collections import defaultdict
from pathlib import Path

NIST, QASPER, NACA = "nist-sp-800-63b-4", "qasper-2105.03011v1", "naca-rm-l56l14a"

# One question per line, so a diff shows exactly which question changed.
# fmt: off
Q = [
    # --- NIST SP 800-63B-4 ------------------------------------------------------------------
    ("nist-01", "What is the lowest assurance level federal agencies may use when they put personal information online?", NIST, [15], ["federal agencies SHALL select a minimum of AAL2 when personal information is made available online"], "AAL2", "paraphrase"),
    ("nist-02", "Does checking for signs of fraud during login change the assurance level of a transaction?", NIST, [15], ["does not impact or change the AAL of a transaction or substitute for an authentication factor"], "No; fraud indicators neither change the AAL nor replace a factor.", "paraphrase"),
    ("nist-03", "To which FIPS 140 level must cryptography in federal verifiers at AAL1 be validated?", NIST, [16, 22], ["Cryptography used by verifiers operated by or on behalf of federal agencies at AAL1 SHALL be validated to meet the requirements of [FIPS140] Level 1.", "FIPS 140 Validation (Government Verifiers and Authenticators) Verifiers –Level 1"], "FIPS 140 Level 1", "lexical"),
    ("nist-04", "How long may a session at the lowest assurance level last before the user has to log in again?", NIST, [17, 22], ["SHOULD be no more than 30 days at AAL1", "Reauthentication (recommended) 30 days overall"], "No more than 30 days (recommended).", "paraphrase"),
    ("nist-05", "What is the recommended inactivity timeout at AAL2?", NIST, [18, 22], ["The inactivity timeout SHOULD be no more than 1 hour.", "24 hours overall 1 hour inactivity"], "No more than 1 hour.", "lexical"),
    ("nist-06", "Is a fingerprint on its own enough to count as an authenticator?", NIST, [18, 19], ["A biometric characteristic is not recognized as an authenticator"], "No; a biometric must be used with a physical authenticator.", "paraphrase"),
    ("nist-07", "Must verifiers offer at least one phishing-resistant authentication option at AAL2?", NIST, [18], ["Verifiers SHALL offer at least one phishing-resistant authentication option at AAL2"], "Yes.", "lexical"),
    ("nist-08", "Can passkeys that sync between devices be used at the highest assurance level?", NIST, [20], ["syncable authenticators SHALL NOT be used at AAL3"], "No; syncable authenticators are not allowed at AAL3.", "paraphrase"),
    ("nist-09", "How long can an AAL3 session sit idle before it should time out?", NIST, [20, 22], ["The inactivity timeout SHOULD be no more than 15 minutes.", "12 hours overall 15 minutes inactivity"], "No more than 15 minutes.", "paraphrase"),
    ("nist-10", "What is the overall timeout for reauthentication at AAL3?", NIST, [20, 22], ["the overall timeout for reauthentication SHALL be no more than 12 hours", "12 hours overall 15 minutes inactivity"], "No more than 12 hours.", "lexical"),
    ("nist-11", "Which baseline of SP 800-53 security controls must a verifier apply?", NIST, [20], ["moderate baseline security controls defined in [SP800-53]"], "The moderate baseline (or an equivalent standard).", "lexical"),
    ("nist-12", "According to the summary table, at which assurance level is exporting keys prohibited?", NIST, [22], ["Key Exportability Permitted Permitted Prohibited"], "AAL3", "table"),
    ("nist-13", "Does the summary of requirements say replay resistance is needed at AAL1?", NIST, [22], ["Replay Resistance Not required Required Required"], "No; it is required only at AAL2 and AAL3.", "table"),
    ("nist-14", "What is the shortest password allowed when it is the only factor used to log in?", NIST, [25], ["to be a minimum of 15 characters in length"], "15 characters.", "paraphrase"),
    ("nist-15", "How short may a password be if it is only ever used together with a second factor?", NIST, [25], ["SHALL require them to be a minimum of eight characters in length"], "At least eight characters.", "paraphrase"),
    ("nist-16", "Should people be forced to change their passwords every few months?", NIST, [25], ["Verifiers and CSPs SHALL NOT require subscribers to change passwords periodically."], "No; only when there is evidence of compromise.", "paraphrase"),
    ("nist-17", "May a site ask \"What was the name of your first pet?\" when someone sets a password?", NIST, [25], ["SHALL NOT prompt subscribers to use knowledge-based authentication (KBA)"], "No; knowledge-based prompts and security questions are not allowed.", "lexical"),
    ("nist-18", "Which Unicode normalization form should be applied to passwords before hashing?", NIST, [26], ["Normalization Form Canonical Composition (NFC)"], "NFC", "lexical"),
    ("nist-19", "Do verifiers have to let people use password managers and autofill?", NIST, [26], ["Verifiers SHALL allow the use of password managers and autofill functionality."], "Yes.", "paraphrase"),
    ("nist-20", "Is there much security gain from making the list of banned passwords enormous?", NIST, [26], ["Excessively large blocklists are of little incremental security benefit"], "No; online guessing is already limited by throttling.", "paraphrase"),
    ("nist-21", "What is the minimum length of a look-up secret?", NIST, [28], ["Look-up secrets SHALL be at least six decimal digits (or equivalent) in length."], "Six decimal digits or equivalent.", "lexical"),
    ("nist-22", "How many times can one code from a printed list of recovery codes be used successfully?", NIST, [28], ["SHALL be used successfully only once"], "Once.", "paraphrase"),
    ("nist-23", "How long must the salt be when short look-up secrets are stored?", NIST, [29], ["The salt value SHALL be at least 32 bits in length"], "At least 32 bits.", "lexical"),
    ("nist-24", "Does receiving a login code by text message protect against phishing?", NIST, [29], ["Out-of-band authentication is not phishing-resistant."], "No.", "paraphrase"),
    ("nist-25", "Should a locked phone show the login code it just received?", NIST, [32], ["the device SHOULD NOT display the authentication secret while it is locked by the owner"], "No; it should only indicate that a secret arrived.", "paraphrase"),
    ("nist-26", "How long does someone have to use a code sent to their phone before the attempt is invalid?", NIST, [33], ["the authentication SHALL be considered invalid unless completed within 10 minutes"], "10 minutes.", "paraphrase"),
    ("nist-27", "Does asking for a new code reset the count of failed attempts?", NIST, [33], ["Generating a new authentication secret SHALL NOT reset the failed authentication count."], "No.", "paraphrase"),
    ("nist-28", "What warning signs should be checked before a code is sent over the phone network?", NIST, [33], ["device swap, SIM change, number porting"], "Risk indicators such as device swap, SIM change and number porting.", "paraphrase"),
    ("nist-29", "Do OTP authenticators and verifiers need FIPS 140 validation?", NIST, [34, 36], ["[FIPS140] validation of OTP authenticators and verifiers is not required."], "No.", "lexical"),
    ("nist-30", "After how many wrong guesses in a row must an authenticator be disabled?", NIST, [41], ["to no more than 100 by disabling that authenticator"], "At most 100.", "paraphrase"),
    ("nist-31", "Who has to tell subscribers how to protect their authenticator from theft or loss?", NIST, [41], ["CSPs SHALL provide subscriber instructions for appropriately protecting the authenticator against theft or loss."], "The CSP.", "paraphrase"),
    ("nist-32", "Should earlier failed attempts be forgotten once a login succeeds?", NIST, [42], ["the verifier SHOULD disregard any previous failed attempts for the authenticators used in the successful authentication"], "Yes, for the authenticators used in the successful login.", "paraphrase"),
    ("nist-33", "What false match rate must a biometric system achieve?", NIST, [43], ["of one in 10000 or better for all demographic groups"], "One in 10,000 or better, for all demographic groups.", "lexical"),
    ("nist-34", "Can I log in by speaking a phrase that is matched against my voice?", NIST, [43], ["Biometric comparison based on voice SHALL NOT be used."], "No; voice biometrics are not allowed.", "paraphrase"),
    ("nist-35", "Must a non-biometric authentication option always be available?", NIST, [43], ["An alternative non-biometric authentication option SHALL always be provided to the subscriber."], "Yes.", "lexical"),
    ("nist-36", "How long must a biometric authenticator wait between tries once the first limit of failures is hit?", NIST, [44], ["impose a delay of at least 30 seconds before each subsequent attempt"], "At least 30 seconds.", "paraphrase"),
    ("nist-37", "Where should biometric matching preferably take place, on the device or on a server?", NIST, [44], ["comparison SHOULD be performed locally"], "Locally, on the device.", "paraphrase"),
    ("nist-38", "How many failed tries in a row are allowed for biometrics, compared with the general limit for authenticators?", NIST, [41, 44], ["to no more than 100 by disabling that authenticator", "SHALL allow no more than five consecutive failed authentication"], "Biometrics: 5 (10 with PAD); authenticators in general: 100.", "multi-page"),
    ("nist-39", "Which web standard used by FIDO2 security keys resists phishing by binding to the site's domain name?", NIST, [47], ["WebAuthn [WebAuthn], which is used by authenticators that implement the Fast"], "WebAuthn", "paraphrase"),
    ("nist-40", "Are passwords replay-resistant?", NIST, [47], ["passwords are not considered replay-resistant"], "No.", "lexical"),
    ("nist-41", "Which smart cards get their phishing resistance from channel binding?", NIST, [47], ["Personal identity verification (PIV) and common access"], "PIV and CAC cards.", "lexical"),
    ("nist-42", "Does looking into the phone's front camera show that I meant to log in?", NIST, [48], ["does not necessarily constitute intent"], "Not necessarily; an explicit action such as tapping a button is needed.", "paraphrase"),
    ("nist-43", "What is the one restricted authenticator at the time these guidelines were published?", NIST, [48], ["there is one restricted authenticator: the use of the PSTN for out-of-band authentication"], "Out-of-band authentication over the PSTN.", "lexical"),
    ("nist-44", "Over what distance may the wireless signal used to set up a hybrid connection reach?", NIST, [52], ["maximum effective range of no more than 240 meters"], "No more than 240 meters.", "paraphrase"),
    ("nist-45", "Where must keys be kept for them to count as non-exportable?", NIST, [52], ["generates, stores, and uses the keys in a protected hardware environment that"], "In protected hardware that software cannot read, such as a TPM or a security key.", "paraphrase"),
    ("nist-46", "How many separate ways of logging in should subscribers keep so they rarely need account recovery?", NIST, [55], ["subscribers to maintain at least two separate means of authentication"], "At least two.", "paraphrase"),
    ("nist-47", "Is replacing a forgotten password, while other authenticators still work, treated as account recovery?", NIST, [58], ["is considered to be the binding of a new authenticator (see Sec. 4.1.2.1) rather than account recovery"], "No; it is treated as binding a new authenticator.", "paraphrase"),
    ("nist-48", "How much randomness must a saved recovery code contain?", NIST, [58], ["The recovery code SHALL include at least 64 bits from an approved"], "At least 64 bits from an approved random bit generator.", "paraphrase"),
    ("nist-49", "How long is a recovery code sent by email valid?", NIST, [59], ["24 hours when sent to an email address"], "24 hours.", "lexical"),
    ("nist-50", "How long does a recovery code mailed inside the contiguous United States stay valid?", NIST, [59], ["21 days when sent to a postal address within the contiguous United States"], "21 days.", "lexical"),
    ("nist-51", "How long is a code sent by text valid for logging in, compared with one sent by text for account recovery?", NIST, [33, 59], ["the authentication SHALL be considered invalid unless completed within 10 minutes", "10 minutes when sent via text message or voice"], "10 minutes in both cases.", "multi-page"),
    ("nist-52", "Can an authenticator still be used after it expires?", NIST, [62], ["When an authenticator expires, it SHALL NOT be usable for authentication."], "No.", "paraphrase"),
    ("nist-53", "Should session tokens survive a restart of the app or a reboot of the device?", NIST, [64], ["Session secrets that are used as bearer tokens for session management SHOULD NOT be persistent"], "No; bearer session secrets should not be persistent.", "paraphrase"),
    ("nist-54", "Which SameSite settings should session cookies use?", NIST, [66], ["SHOULD set “SameSite=Lax” or “SameSite=Strict”."], "SameSite=Lax or SameSite=Strict.", "lexical"),
    ("nist-55", "Does holding an OAuth access token prove that the subscriber is still there?", NIST, [66], ["The RP SHALL NOT interpret the presence of an access token as an indicator of the subscriber’s presence"], "No.", "lexical"),
    ("nist-56", "May a session cookie hold someone's personal details in plain text?", NIST, [66], ["SHALL NOT contain cleartext personal information"], "No; it should contain only an opaque string.", "paraphrase"),
    ("nist-57", "Does activity during a session reset the overall timeout?", NIST, [67], ["Session activity SHALL reset the inactivity timeout, and successful reauthentication during a session SHALL reset both"], "No; activity resets only the inactivity timeout, and reauthentication resets both.", "paraphrase"),
    ("nist-58", "What do the guidelines call watching a session's behaviour for signs of fraud?", NIST, [68], ["Session monitoring (sometimes called continuous authentication)"], "Session monitoring, sometimes called continuous authentication.", "paraphrase"),
    ("nist-59", "Can someone's typing rhythm be watched during a session to spot fraud?", NIST, [68], ["Behavioral biometric characteristics (e.g., typing cadence)"], "Yes; it is one of the characteristics that may be evaluated.", "paraphrase"),
    # --- Qasper paper --------------------------------------------------------------------------
    ("qasper-01", "How many questions and papers does the dataset contain?", QASPER, [1, 2], ["5,049 questions over 1,585"], "5,049 questions over 1,585 papers.", "paraphrase"),
    ("qasper-02", "How many questions does each paper have on average?", QASPER, [2], ["Each paper has an average of 3.2 questions"], "3.2", "paraphrase"),
    ("qasper-03", "What share of the questions need evidence from more than one paragraph?", QASPER, [2], ["55.5% of the questions require evidence from multiple paragraphs"], "55.5%", "paraphrase"),
    ("qasper-04", "What did the workers earn per hour on average?", QASPER, [2], ["US$25 per hour on average"], "US$25 per hour on average ($20–$40 by experience).", "paraphrase"),
    ("qasper-05", "How many people wrote questions, and how many answered them?", QASPER, [2], ["A total of 25 workers wrote questions while 51 answered them."], "25 wrote questions and 51 answered them.", "paraphrase"),
    ("qasper-06", "Which collection of open-access papers was filtered to build the dataset?", QASPER, [2], ["We filtered S2ORC (Lo et al., 2020)"], "S2ORC", "paraphrase"),
    ("qasper-07", "What share of the question writers had more than five years of experience?", QASPER, [3], ["with 20% having more than five years of experience"], "20%", "lexical"),
    ("qasper-08", "Which tool was used to build the tutorials and qualification exams for the answer annotators?", QASPER, [3], ["using CrowdAQ (Ning et al., 2020)"], "CrowdAQ", "paraphrase"),
    ("qasper-09", "How well did the two experts agree when they labelled question types?", QASPER, [3], ["achieved a Cohen’sκ of 0.94"], "Cohen's kappa of 0.94.", "paraphrase"),
    ("qasper-10", "How long are extractive answers on average?", QASPER, [3], ["The average length of the extractive answers is 14.4 words"], "14.4 words.", "lexical"),
    ("qasper-11", "In the table of examples, what share of questions are specific to their paper?", QASPER, [4], ["What other political events are included in the database? Specific 66.7%"], "66.7%", "table"),
    ("qasper-12", "How many annotators answered each question on average?", QASPER, [4], ["each question is answered by 1.6 annotators"], "1.6", "paraphrase"),
    ("qasper-13", "What share of the sampled answers were judged correct?", QASPER, [4], ["We found that 207 (75.8%) of the answers were correct."], "207 of them, 75.8%.", "paraphrase"),
    ("qasper-14", "How many questions are in the train, validation and test sets?", QASPER, [5], ["This resulted in 2,593, 1,005, and 1,451 questions"], "2,593 train, 1,005 validation and 1,451 test.", "paraphrase"),
    ("qasper-15", "What is the estimated human performance in Answer-F1 and Evidence-F1?", QASPER, [5], ["estimates the human performance to be 60.9 Answer-F1, and 71.6 Evidence-F1"], "60.9 Answer-F1 and 71.6 Evidence-F1.", "lexical"),
    ("qasper-16", "Which model reads the long papers in the baseline system?", QASPER, [5], ["We use the Longformer-Encoder-Decoder (LED; Beltagy et al., 2020)"], "The Longformer-Encoder-Decoder (LED).", "paraphrase"),
    ("qasper-17", "What share of the papers fit into the model's 16K-token input?", QASPER, [5], ["we can encode 99% of the paper full texts in the QASPER dataset"], "99%", "paraphrase"),
    ("qasper-18", "Which GPU were the experiments run on?", QASPER, [6], ["single RTX 8000 GPU"], "A single RTX 8000 GPU.", "lexical"),
    ("qasper-19", "How much warmup did the learning rate scheduler use?", QASPER, [6], ["with 10% warmup"], "10% warmup.", "lexical"),
    ("qasper-20", "What test Answer-F1 did the full-text model with the evidence scaffold reach?", QASPER, [7], ["Q+Full Text w/ scaff. 24.62 29.97 13.86 15.02 63.64 68.90 38.89 44.97 28.01 33.63"], "33.63 overall on test.", "table"),
    ("qasper-21", "How did the TF-IDF baseline do at selecting evidence on the test set?", QASPER, [7], ["TF-IDF 10.68 9.20"], "9.20 Evidence-F1.", "table"),
    ("qasper-22", "How many test questions with very low scores were sampled for the error analysis?", QASPER, [7], ["we sample 67 test questions with predicted"], "67", "paraphrase"),
    ("qasper-23", "By how much are the model's predicted spans shorter than the gold answers?", QASPER, [7], ["9.5 words shorter than gold counterparts, on average"], "9.5 words shorter on average.", "paraphrase"),
    ("qasper-24", "Which model answers best when it is given the gold evidence?", QASPER, [8], ["UnifiedQA-large 67.23 28.92 61.39"], "UnifiedQA-large, 61.39 overall Answer-F1.", "table"),
    ("qasper-25", "How many questions do PubmedQA and BioAsq contain?", QASPER, [8], ["PubmedQA and BioAsq contain 1K and 3.2K questions, respectively"], "1K and 3.2K.", "paraphrase"),
    ("qasper-26", "How do human lower bounds compare with the best model, both at answering and at selecting evidence?", QASPER, [5, 7], ["estimates the human performance to be 60.9 Answer-F1, and 71.6 Evidence-F1", "LED-large 31.25 39.37"], "Humans 60.9 Answer-F1 and 71.6 Evidence-F1; the best evidence model (LED-large) reaches 39.37.", "table"),
    # --- NACA RM L56L14a (a scan with an OCR layer) --------------------------------------------
    ("naca-01", "When was the memorandum on the 10-degree sweptback wing declassified?", NACA, [1], ["Declassified March 16, 1959"], "March 16, 1959.", "scanned"),
    ("naca-02", "Who wrote the report on the transonic flutter of the thin sweptback wing?", NACA, [1, 2], ["By George W. Jones, Jr.", "ByGeorgeW. fones, Jr."], "George W. Jones, Jr.", "scanned"),
    ("naca-03", "In which wind tunnel was the flutter investigation carried out?", NACA, [2, 3, 6], ["A flutter investigation has been made in the Langley transonic blowdown tunnel", "investigation was undertaken in the Langley transonic", "The tests were made in the Langley transonic blowdown tunnel"], "The Langley transonic blowdown tunnel.", "scanned"),
    ("naca-04", "How does the flutter boundary near Mach 0.8 for the 10-degree wing compare with the other sweep angles?", NACA, [2], ["subsonic level (around a Mach number of 0.8) of the flutter boundary"], "It lies above those of the 0° and 30° sweptback wings.", "scanned"),
    ("naca-05", "What material were the flutter models built from?", NACA, [5], ["a phenolic laminate material with high-strength paper reinforcement"], "Solid Consoweld, a phenolic laminate with paper reinforcement.", "scanned"),
    ("naca-06", "How wide is the tunnel's test section?", NACA, [6], ["octagonal test section measuring 26 inches between flats"], "26 inches between flats (octagonal).", "scanned"),
    ("naca-07", "What range of air density can the tunnel produce?", NACA, [6], ["from approximately 0.001 to 0.012 slug per cubic foot"], "About 0.001 to 0.012 slug per cubic foot.", "scanned"),
    ("naca-08", "How much does the model support system weigh?", NACA, [6], ["its weight is 289 pounds"], "289 pounds.", "scanned"),
    ("naca-09", "What is a period of intermittent oscillations before continuous flutter called?", NACA, [8], ["Such periods are designated low-damping regions"], "A low-damping region.", "scanned"),
    ("naca-10", "What was used to record the strain-gage signals continuously?", NACA, [6], ["A recording oscillograph was used to give"], "A recording oscillograph.", "scanned"),
    # --- unanswerable: plausible questions the corpus does not answer ----------------------------
    ("un-01", "What is the recommended inactivity timeout at AAL4?", NIST, [], [], None, "lexical"),
    ("un-02", "How much does a FIDO2 security key cost?", NIST, [], [], None, "paraphrase"),
    ("un-03", "What is the maximum length allowed for a username?", NIST, [], [], None, "paraphrase"),
    ("un-04", "Which countries have made SP 800-63B-4 law?", NIST, [], [], None, "paraphrase"),
    ("un-05", "May authentication codes be sent by fax?", NIST, [], [], None, "lexical"),
    ("un-06", "How many hours did the annotators spend writing questions in total?", QASPER, [], [], None, "paraphrase"),
    ("un-07", "How many Qasper papers are about computer vision?", QASPER, [], [], None, "lexical"),
    ("un-08", "Which company sponsored the GPUs used for training?", QASPER, [], [], None, "paraphrase"),
    ("un-09", "What accuracy does GPT-3 reach on Qasper?", QASPER, [], [], None, "lexical"),
    ("un-10", "What was the maximum lift coefficient measured on the wing?", NACA, [], [], None, "scanned"),
    ("un-11", "How much did the flutter tests cost?", NACA, [], [], None, "scanned"),
    ("un-12", "How many passwords does the average person reuse across sites?", NIST, [], [], None, "paraphrase"),
    ("un-13", "Which programming language was the LED model written in?", QASPER, [], [], None, "paraphrase"),
    ("un-14", "At what altitude was the wing flown in free flight?", NACA, [], [], None, "scanned"),
    ("un-15", "What is the penalty for an agency that ignores these guidelines?", NIST, [], [], None, "paraphrase"),
    ("un-16", "Which annotator wrote the most questions?", QASPER, [], [], None, "paraphrase"),
]
# fmt: on


def records() -> list[dict[str, object]]:
    out = []
    for qid, question, doc, pages, evidence, answer, kind in Q:
        out.append(
            {
                "id": qid,
                "question": question,
                "lang": "en",
                "doc": doc,
                "pages": pages,
                "evidence": evidence,
                "answer": answer,
                "kind": kind,
                "answerable": bool(pages),
            }
        )
    return out


def split(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(row["doc"], row["kind"], row["answerable"])].append(row)
    out: dict[str, list[dict[str, object]]] = {"dev": [], "test": []}
    for key in sorted(groups, key=str):
        for i, row in enumerate(sorted(groups[key], key=lambda r: str(r["id"]))):
            out["dev" if i % 2 == 0 else "test"].append(row)
    return {name: sorted(rows, key=lambda r: str(r["id"])) for name, rows in out.items()}


def main() -> None:
    folder = Path("eval/questions")
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in split(records()).items():
        lines = [json.dumps(r, ensure_ascii=False) for r in rows]
        (folder / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(name, len(rows))


if __name__ == "__main__":
    main()
