"""exp3 generative math eval — faithful replication of lm-eval-harness tasks.

  - gsm8k   : lm-eval `gsm8k_cot` (lm_eval/tasks/gsm8k/gsm8k-cot.yaml, v3.0).
              8-shot CoT with the VERBATIM canonical exemplars (incl. the two
              "9 + 20 is 29" / "23 - 15 is 8" quirks), "Q: ...\nA:" format,
              stops ["Q:", "</s>", "<|im_end|>"]. Scoring = both lm-eval
              filters: strict-match (regex `The answer is (\\-?[0-9\\.\\,]+).`,
              FIRST match) as the primary metric, flexible-extract (last
              number-ish match) as a secondary metric; exact_match comparison
              with ignore_case + regexes_to_ignore [",", "\\$",
              "(?s).*#### ", "\\.$"], no-match -> "[invalid]".
  - math500 : lm-eval `minerva_math500` (minerva_math_algebra.yaml + dataset
              HuggingFaceH4/MATH-500; lm_eval/tasks/minerva_math/utils.py).
              4-shot Minerva prompt with VERBATIM exemplars, "Problem:\n...\n
              \nSolution:" format joined with a single space before the
              solution, stop ["Problem:"]. Gold = normalize_final_answer(
              remove_boxed(last_boxed_only_string(solution))) — from the
              SOLUTION field, not the answer field. Scoring = get_unnormalized_
              answer ("Final Answer: The final answer is ... I hope it is
              correct." — the Minerva phrase, NOT \\boxed) -> normalize_final_
              answer -> is_equiv (sympy parse_latex). One deliberate deviation:
              we accept normalized string equality BEFORE is_equiv (upstream is
              sympy-only, which false-negatives unparseable-but-identical
              answers like "[2,5)"). Optional `math_verify` metric (same as
              upstream's second metric) when the math_verify package is
              importable.

Generation is run elsewhere (run.py); here we only build prompts and score.
`stop` is a list of strings; score() truncates the generation at the earliest
occurrence of any of them (lm-eval `until` semantics).
"""
from __future__ import annotations

import logging
import re
import signal

eval_logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# gsm8k — lm-eval gsm8k_cot: 8-shot CoT exemplars, VERBATIM from
# lm_eval/tasks/gsm8k/gsm8k-cot.yaml (fewshot_config.samples). Do NOT "fix"
# the two arithmetic-phrasing quirks ("9 + 20 is 29", "23 - 15 is 8") — they
# are part of the canonical Wei et al. prompt as shipped by lm-eval.
# --------------------------------------------------------------------------
_GSM8K_8SHOT = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
     "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. "
     "So there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
     "After eating 35, they had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
     "How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
     "So he gave Denny 20 - 12 = 8. The answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
     "How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. "
     "5 + 4 = 9. The answer is 9."),
    ("There were nine computers in the server room. Five more computers were installed each day, "
     "from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. "
     "So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
     "How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
     "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
     "So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."),
]

# lm-eval prompt assembly: doc_to_text "Q: {question}\nA:" + target_delimiter
# " " + target, examples joined by fewshot_delimiter "\n\n".
_GSM8K_STOPS = ["Q:", "</s>", "<|im_end|>"]


def _gsm8k_fewshot_prefix(num_fewshot: int) -> str:
    shots = _GSM8K_8SHOT[:num_fewshot]
    return "".join(f"Q: {q}\nA: {a}\n\n" for q, a in shots)


# --------------------------------------------------------------------------
# MATH-500 — lm-eval minerva_math500: 4-shot Minerva prompt, VERBATIM from
# lm_eval/tasks/minerva_math/utils.py list_fewshot_samples() (byte-for-byte,
# including the stray "}" in exemplar 1, double spaces, and the single-
# backslash line breaks inside align* blocks).
# --------------------------------------------------------------------------
_MATH_4SHOT = [
    ("Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
     "The expressions inside each square root must be non-negative. Therefore, "
     "$x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator "
     "cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of "
     "the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct."),
    ("If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
     "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = "
     "\\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct."),
    ("Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, "
     "how many times must Terrell lift them in order to lift the same total weight?",
     "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds "
     "of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of "
     "$2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n"
     "\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\n"
     "Final Answer: The final answer is $16$. I hope it is correct."),
    ("If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution "
     "$(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
     "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$"
     "Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}="
     "\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct."),
]

_MATH_STOPS = ["Problem:"]


def _math_fewshot_prefix(num_fewshot: int) -> str:
    shots = _MATH_4SHOT[:num_fewshot]
    # doc_to_text "Problem:\n{problem}\n\nSolution:" + target_delimiter " " + solution
    return "".join(f"Problem:\n{p}\n\nSolution: {s}\n\n" for p, s in shots)


# --------------------------------------------------------------------------
# loaders -> list of dicts {prompt, gold, stop, meta}
# --------------------------------------------------------------------------
def load_dataset_prompts(name: str, num_fewshot: int | None, limit: int) -> list[dict]:
    """Return list of {"prompt", "gold", "stop", "meta"}. limit<=0 -> all."""
    from datasets import load_dataset

    if name == "gsm8k":
        nfs = 8 if num_fewshot is None else num_fewshot
        prefix = _gsm8k_fewshot_prefix(nfs)
        ds = load_dataset("openai/gsm8k", "main", split="test")
        rows = []
        for ex in ds:
            # lm-eval doc_to_target: answer.split('####')[-1].strip()
            gold = ex["answer"].split("####")[-1].strip()
            rows.append({
                "prompt": prefix + f"Q: {ex['question']}\nA:",
                "gold": gold,
                "stop": _GSM8K_STOPS,
                "meta": {"question": ex["question"]},
            })
            if 0 < limit <= len(rows):
                break
        return rows

    if name == "math500":
        nfs = 4 if num_fewshot is None else num_fewshot
        prefix = _math_fewshot_prefix(nfs)
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        rows = []
        for ex in ds:
            # lm-eval process_docs: gold from the SOLUTION's last \boxed{...}
            boxed = last_boxed_only_string(ex["solution"])
            raw = remove_boxed(boxed) if boxed is not None else ex["answer"]
            rows.append({
                "prompt": prefix + f"Problem:\n{ex['problem']}\n\nSolution:",
                "gold": normalize_final_answer(raw if raw is not None else ex["answer"]),
                "stop": _MATH_STOPS,
                # solution kept for the optional math_verify metric (gold side).
                "meta": {"subject": ex.get("subject"), "level": ex.get("level"),
                         "solution": ex["solution"]},
            })
            if 0 < limit <= len(rows):
                break
        return rows

    raise ValueError(f"Unknown dataset: {name} (use gsm8k | math500)")


def score(name: str, generation: str, gold: str, stop, meta: dict | None = None,
          ) -> tuple[bool, str, dict]:
    """Truncate generation at the earliest stop string, extract the predicted
    answer, compare to gold. Returns (correct, predicted_str, extras) where
    `correct` is the task's primary lm-eval metric (gsm8k: strict-match;
    math500: minerva exact_match) and `extras` holds secondary metrics
    (gsm8k: correct_flex/pred_flex; math500: math_verify or None)."""
    generation = _truncate_at_stops(generation, stop)
    if name == "gsm8k":
        pred_s = _gsm8k_extract_strict(generation)
        pred_f = _gsm8k_extract_flexible(generation)
        return (_gsm8k_exact_match(pred_s, gold), pred_s,
                {"correct_flex": _gsm8k_exact_match(pred_f, gold), "pred_flex": pred_f})
    if name == "math500":
        pred = normalize_final_answer(get_unnormalized_answer(generation))
        ok = (pred != "[invalidanswer]") and (pred == gold or is_equiv(pred, gold))
        mv = _math_verify_metric(generation, (meta or {}).get("solution"))
        return ok, pred, {"math_verify": mv}
    raise ValueError(name)


def _truncate_at_stops(text: str, stop) -> str:
    if stop is None:
        return text
    stops = [stop] if isinstance(stop, str) else stop
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i >= 0:
            cut = min(cut, i)
    return text[:cut]


# --------------------------------------------------------------------------
# gsm8k scoring — lm-eval gsm8k_cot filters + exact_match, replicated.
# --------------------------------------------------------------------------
_GSM8K_STRICT_RE = re.compile(r"The answer is (\-?[0-9\.\,]+).")
_GSM8K_FLEX_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
# evaluate.exact_match regexes_to_ignore (applied to BOTH pred and gold):
_GSM8K_IGNORE_RES = [",", r"\$", r"(?s).*#### ", r"\.$"]
_INVALID = "[invalid]"


def _gsm8k_extract_strict(text: str) -> str:
    # RegexFilter default: findall + take_first (FIRST occurrence).
    m = _GSM8K_STRICT_RE.findall(text)
    return m[0] if m else _INVALID


def _gsm8k_extract_flexible(text: str) -> str:
    # group_select=-1: LAST regex match; tuple -> first non-empty group.
    ms = _GSM8K_FLEX_RE.findall(text)
    if not ms:
        return _INVALID
    last = ms[-1]
    return next((g for g in last if g), _INVALID)


def _gsm8k_exact_match(pred: str, gold: str) -> bool:
    for rgx in _GSM8K_IGNORE_RES:
        pred = re.sub(rgx, "", pred)
        gold = re.sub(rgx, "", gold)
    return pred.lower() == gold.lower()   # ignore_case=True, no punctuation strip


# --------------------------------------------------------------------------
# MATH scoring — lm-eval minerva_math utils.py, replicated (boxed helpers,
# Minerva answer normalization from Lewkowycz et al. (2022) appendix D,
# sympy-based is_equiv).
# --------------------------------------------------------------------------
def last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str | None:
    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] != left:
            return None
        return s[len(left):]
    left = "\\boxed{"
    if not (s[: len(left)] == left and s[-1] == "}"):   # upstream asserts; we soft-fail
        return None
    return s[len(left) : -1]


class _timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    """x1 and x2 are normalized latex strings (sympy equivalence, 5s timeout)."""
    try:
        import sympy
        from sympy.parsing.latex import parse_latex
    except ImportError as e:
        eval_logger.warning(f"sympy/antlr unavailable ({e}); is_equiv -> False")
        return False
    try:
        with _timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except Exception:
                eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False
            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                return False
            try:
                return sympy.simplify(diff) == 0
            except ValueError:
                return False
    except TimeoutError:
        return False
    except Exception:
        return False


def get_unnormalized_answer(text: str) -> str:
    INVALID_ANSWER = "[invalidanswer]"
    end_seq = "I hope it is correct."
    text += end_seq
    match = re.search(r"Final Answer: The final answer is(.*?). I hope it is correct.", text)
    return match.group(1).strip() if match else INVALID_ANSWER


SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""), (" ", ""),
    ("mbox", "text"), (",\\text{and}", ","), ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "ft", "hours",
    "km", "units", "\\ldots", "sue", "points", "feet", "minutes", "digits",
    "cents", "degrees", "cm", "gm", "pounds", "meters", "meals", "edges",
    "students", "childrentickets", "multiples", "\\text{s}", "\\text{.}",
    "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}",
    r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Copied character for character from appendix D of Lewkowycz et al. (2022)."""
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer


def _math_verify_metric(candidates: str, gold_solution: str | None):
    """Upstream's second metric: math_verify.verify(parse(solution), parse(gen)).
    Returns True/False, or None when math_verify (or the solution) is missing."""
    if gold_solution is None:
        return None
    try:
        from math_verify import parse, verify
    except ImportError:
        return None
    try:
        return bool(verify(gold=parse(gold_solution), target=parse(candidates)))
    except Exception:
        return False
