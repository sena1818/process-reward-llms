# Speaker Script: Let's Verify Step by Step
## Process Reward in Large Language Models Seminar | 2026-04-27

---

> Note: text in [brackets] is a stage cue and should not be read aloud.
> Estimated speaking time: about 20 to 22 minutes. With pauses, slide changes, and emphasis, this fits a standard seminar talk well.

---

## Slide 1. Opening

The paper I am presenting today is *Let's Verify Step by Step*, by Hunter Lightman and collaborators from OpenAI, first released in 2023 and later published at ICLR 2024.

I think this paper is one of the most representative papers for a course called *Process Reward in Large Language Models*, because it asks a very direct question:

When we train language models for difficult reasoning, should we reward only the final answer, or should we reward the reasoning process itself?

The central claim of the paper is very clear:

**For complex mathematical reasoning, step-level process supervision works better than supervision based only on the final outcome.**

So the contribution of the paper is not only a higher benchmark number. It is really a shift in what we choose to supervise.

---

## Slide 2. Why This Matters for Process Reward

Before getting into the method, I want to frame why this question matters.

Large language models can now generate step-by-step solutions that look very persuasive. But looking like reasoning is not the same thing as being correct reasoning.

A model may make a subtle mistake in the middle of a solution, and from that point on the entire chain is broken.

There is also a second, more interesting failure mode:

the model can use flawed reasoning and still end up with the correct final answer by accident.

If we only reward the final answer, then both of these cases are hard to distinguish from genuinely good reasoning.

So from a process reward perspective, the paper is really about this:

**If the target behavior is reliable reasoning, then the reward signal should say something about the reasoning process, not only about the endpoint.**

---

## Slide 3. Outcome Supervision vs Process Supervision

The paper compares two supervision strategies.

The first is **outcome supervision**.

Here, the reward model only sees whether the final answer is correct or incorrect.

This is simple and cheap, because in mathematics the final answer can often be checked automatically.

But the weakness is that the signal is extremely coarse.

If a full solution is wrong, the model only learns that something went wrong somewhere. It does not learn where the first mistake happened.

The second strategy is **process supervision**.

Here, human annotators inspect the intermediate reasoning steps and label whether each step is acceptable.

This gives much denser feedback, and it gives much better credit assignment.

So the key comparison in the paper is not between two generators. It is between two ways of defining what counts as a good solution.

---

## Slide 4. Method Overview

The overall experimental setup is conceptually simple.

First, a generator model produces many candidate solutions for each math problem.

Then the authors train two different reward models:

- an ORM, or outcome-supervised reward model
- and a PRM, or process-supervised reward model

At test time, the generator produces many candidate solutions again, and the reward model selects the one it believes is best.

So the evaluation is based on **best-of-N search**:

out of many candidate solutions, which reward model is better at identifying the truly correct one?

This setup is important because it isolates the role of supervision inside the verifier or reward model, rather than mixing everything into one end-to-end generation score.

---

## Slide 5. PRM800K and the Labeling Design

One of the major contributions of the paper is the dataset called **PRM800K**.

This dataset contains roughly:

- 800 thousand step-level labels
- around 75 thousand solutions
- over 12 thousand math problems

The labels are attached to intermediate reasoning steps rather than only to final answers.

A very important design choice is that annotators only label up to the **first incorrect step**.

Why does that matter?

Because it makes the comparison with outcome supervision more conservative and more fair.

For a wrong solution, both approaches know that the solution is wrong overall. The extra information process supervision gets is mainly the location of the mistake.

So even under this relatively conservative setup, the authors still find a clear advantage for process supervision.

That is one reason the paper is persuasive.

---

## Slide 6. Main Result

Now to the central result.

In the large-scale experiment, the authors compare three selection methods:

- majority voting
- the outcome-supervised reward model
- the process-supervised reward model

The headline numbers on their representative MATH evaluation subset are:

- Majority Voting: 69.6 percent
- ORM: 72.4 percent
- PRM: 78.2 percent

So the process-supervised reward model is clearly the strongest of the three.

And the paper emphasizes an even more interesting pattern:

as the number of candidate solutions increases, the gap between PRM and ORM becomes larger.

This suggests that process supervision is especially useful when the model has to search over a large reasoning space and distinguish subtle differences in solution quality.

In other words, PRM is not just slightly better. It is better at ranking candidate chains of thought.

---

## Slide 7. Why Does PRM Win?

The natural question is: why does process supervision help so much?

I think there are three main reasons.

First, **better credit assignment**.

Outcome supervision says only that the whole trajectory succeeded or failed. Process supervision can point to the first bad step.

Second, **denser learning signal**.

Instead of one label for a full solution, the reward model receives information throughout the trajectory.

Third, **less reward for spurious reasoning**.

If a model reaches the correct final answer through a flawed derivation, outcome supervision may still reward that solution.

Process supervision is much more likely to detect that the reasoning path itself is unreliable.

So the deeper lesson is that supervision granularity shapes what the model is actually able to learn about reasoning quality.

---

## Slide 8. Active Learning and Annotation Cost

At this point, an obvious objection is that process supervision sounds expensive.

And that objection is correct.

Human step-by-step annotation is much more costly than checking final answers automatically.

The paper addresses this with **active learning**.

Instead of labeling arbitrary solutions, the annotators focus more on the examples that are most informative or most confusing for the current reward model.

The result is that active learning improves label efficiency by about **2.6 times**.

This does not make process supervision free.

But it does show that the cost problem can be reduced if annotation is targeted rather than uniform.

That is an important practical point, because one common criticism of process reward methods is that they are too expensive to scale.

---

## Slide 9. Generalization and Alignment Meaning

The paper also evaluates out-of-distribution generalization on new STEM-style problems.

There again, the process-supervised model remains stronger:

- ORM: 63.8 percent
- PRM: 72.9 percent

So the advantage is not limited to one exact test set.

The paper also makes a broader alignment argument.

Process supervision is more interpretable, easier for humans to audit, and more closely tied to the behavior we actually want.

The authors describe this as a kind of **negative alignment tax**:

in this case, the more aligned supervision strategy is not weaker. It is actually stronger.

That is a very attractive result, because it suggests that capability and alignment do not always have to be in tension.

---

## Slide 10. Limits and Open Questions

Even though the paper is influential, it also has important limitations.

First, the main evidence comes from mathematical reasoning.

Mathematics is a favorable domain because final answers are often verifiable, and intermediate steps are relatively structured.

It is much less obvious how to apply the same idea in domains like law, medicine, open-ended planning, or creative reasoning.

Second, process labels remain expensive even with active learning.

So a major open problem is whether strong models or automated verifiers can help generate reliable step-level supervision more cheaply.

Third, the final evaluation metric is still outcome-based.

In the end, success is still defined by whether the selected solution solves the problem.

So the paper shows that process supervision is a better way to optimize outcome success, but it does not fully solve the broader philosophical question of what “good reasoning” should mean in every domain.

---

## Slide 11. Closing

To conclude, I think the paper makes three lasting points.

First, if we only supervise final answers, we throw away critical information about where reasoning goes right or wrong.

Second, step-level supervision can produce substantially better reward models for mathematical reasoning.

Third, process reward is not only an interpretability idea. It can also be a performance advantage.

So my one-sentence summary would be:

**This paper moves reward learning from “Was the answer correct?” to “Was the reasoning correct step by step?”, and shows that this finer-grained supervision leads to more reliable reasoning selection.**

Thank you.

---

## Discussion Questions

### Q1

If process supervision works so well in mathematics, what would be the right analogue in less structured domains such as legal reasoning, biomedical reasoning, or long-horizon planning?

### Q2

Is step-level human annotation a scalable long-term solution, or is it mainly a bootstrap method until we have stronger automated verifiers?

### Q3

If outcome-based evaluation remains the final metric, to what extent are we really measuring better reasoning, rather than just better selection for tasks with checkable answers?

---

## Timing Estimate

| Slide group | Content | Estimated time |
|---|---|---|
| 1 to 2 | Opening and motivation | 3 min |
| 3 to 5 | Definitions, framework, PRM800K | 6 min |
| 6 to 7 | Main result and explanation | 5 min |
| 8 to 9 | Active learning, OOD, alignment meaning | 4 min |
| 10 to 11 | Limits, summary, closing | 3 to 4 min |
| Discussion | Questions | flexible |
