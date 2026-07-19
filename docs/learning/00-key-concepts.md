# Key concepts to understand this thesis

This page is for actually *understanding* the ideas behind the project, not
just looking up a term (that's what
[`02-glossary.md`](02-glossary.md) is for). Read this if you want to be
able to explain, in your own words, why the thesis is built the way it is.

## Correlation vs. causation, with a concrete example

Say a dataset shows: accidents happen more often on rainy days. A
correlation-based ML model learns exactly that pattern — "rain → higher
predicted risk" — and gets good at predicting it. But it can't answer a
follow-up question like *"if we made this road less slippery when wet,
would risk actually go down, or is rain just correlated with something
else that's the real cause (like reduced visibility, or fewer cars slowing
down in time)?"*

Causal reasoning tries to answer that second kind of question:
**if I change one specific thing, and hold everything else constant, what
happens to the outcome?** That's called an *intervention*, and asking "what
if X had been different?" is the whole point of causal-style analysis.
Proper causal inference (as in Judea Pearl's work, cited in the thesis) does
this rigorously, using statistics on real data, and can *discover* which
variables actually matter. This thesis does a much simpler, manual version
of the same idea — see the next section.

## Why a hand-built model still counts as "causal-style"

This project doesn't discover anything from data. Instead, a person
(informed by common sense) writes down: "I believe speed, visibility,
proximity, and weather affect accident risk," draws that as a simple
diagram, and hand-picks how much each one should count. Then the
*mechanism* — changing one variable and recomputing the outcome — is
exactly the same mechanism real causal inference uses. Think of it as
borrowing the *reasoning pattern* of causal inference (intervene, observe
the change) without borrowing the *statistical machinery* that would
normally justify it. That's a real limitation, and the thesis says so
explicitly — but it's still a genuinely different and more useful way to
interact with a risk model than a plain prediction score, because you can
always ask "what if?" and get an answer you can trace by hand.

## What a causal graph (DAG) actually is

A DAG (Directed Acyclic Graph) is just boxes and arrows: each box is a
variable, each arrow means "this box influences that box," and "acyclic"
means the arrows never loop back on themselves (nothing causes itself,
directly or indirectly). This thesis's DAG is about as simple as a DAG can
be: four boxes (Weather, Visibility, Speed, Proximity), all with one arrow
each pointing at a fifth box (Accident Risk). No arrows between the four
input variables themselves — they're treated as independent inputs, which
is a simplification (in reality, weather obviously affects visibility too)
made explicitly to keep the model easy to compute and explain.

## Why "interpretable" is the actual selling point

A lot of accident-detection research chases the highest possible accuracy,
usually with deep learning models that are hard to inspect — you can see
their output, but not *why* they produced it. This thesis takes the
opposite trade-off on purpose: lower ceiling on accuracy (since real
accident risk depends on far more than 4 hand-picked variables), in
exchange for a model where every number can be explained in one sentence,
by hand, to a non-technical person. That trade-off — interpretability over
raw predictive power — is the actual research contribution being
demonstrated, more than the specific formula used.

## Where to go next

- [`01-dataset-and-related-work.md`](01-dataset-and-related-work.md) — what
  the VRU-Accident dataset actually contains, and how this thesis compares
  to other accident-analysis research.
- [`../approach/00-approach-overview.md`](../approach/00-approach-overview.md)
  — how these concepts turned into actual code.
