# 2026-04-26 Seminar Prep: *Let's Verify Step by Step*（中文版）

课程：Process Reward in Large Language Models

论文：Hunter Lightman et al., *Let's Verify Step by Step*, OpenAI, 2023

论文链接：

- [OpenAI PDF](https://cdn.openai.com/improving-mathematical-reasoning-with-process-supervision/Lets_Verify_Step_by_Step.pdf)
- [arXiv 2305.20050](https://arxiv.org/abs/2305.20050)

---

## 1. 这篇论文一句话在讲什么

这篇论文想回答一个非常核心的问题：

**训练大语言模型做复杂推理时，我们到底应该只看“最后答案对不对”，还是应该检查“中间每一步推理对不对”？**

作者的核心结论是：

**在复杂数学推理上，按步骤给监督（process supervision）比只看最终结果给监督（outcome supervision）更有效，也更可靠。**

---

## 2. 题目和摘要的大致中文意译

### 2.1 题目怎么翻

`Let's Verify Step by Step`

比较自然的中文可以讲成：

- `让我们一步一步来验证`
- `我们来逐步验证推理过程`

如果你是在 seminar 里口头讲，我建议直接说：

**“这篇论文的核心主张就是：不要只看答案，要一步一步验证模型的推理过程。”**

### 2.2 摘要意译

下面不是逐字翻译，而是更适合你口头表达的意译版本。

随着大语言模型越来越擅长多步推理，它们也仍然会经常在中间某一步犯逻辑错误。为了让模型更可靠，我们可以训练 reward model 来区分“好推理”和“坏推理”。问题在于，监督信号可以有两种给法：

- 一种是 outcome supervision，只告诉模型最后答案对不对；
- 另一种是 process supervision，对中间每一步都给反馈。

这篇论文系统比较了这两种方法，发现在困难数学数据集 MATH 上，process supervision 明显优于 outcome supervision。作者训练出的 process-supervised reward model 在一个代表性的 MATH 测试子集上可以达到 **78.2%** 的 best-of-N 选择效果；同时作者还发现，**active learning** 能显著提高标注效率。最后，他们发布了一个重要数据集 **PRM800K**，里面包含大约 **80 万条 step-level 人类反馈标签**。

---

## 3. 你讲这篇论文时最该抓住的主线

最适合 seminar 的主线不是“作者又做了一个更强的数学模型”，而是下面这句：

**如果奖励只看最终答案，那么模型可能学会“结果看起来对”，却没有真正学会“过程是对的”；而 process reward 的意义，就是把奖励信号从结果推进到推理过程本身。**

你可以把整场报告串成下面这条逻辑链：

1. LLM 会写出 step-by-step reasoning，但中间可能悄悄犯错。
2. 只看 final answer 的监督太粗糙，无法告诉模型“错在第几步”。
3. 所以作者提出：训练 reward model 时，应该去标每一步是否合理。
4. 实验表明，这样训练出来的 reward model 更会挑出真正正确的解法。
5. 这不仅提升性能，也更符合 alignment 里“奖励正确过程”的思路。

---

## 4. 这篇论文为什么经典

它经典，主要不是因为某一个数字高，而是因为它把一个很重要的观念讲清楚了：

- **reward model 不一定只能对最终结果打分**
- **过程本身可以被监督**
- **更细粒度的监督，可能同时提升能力和对齐性**

这篇论文后来影响了很多关于：

- process reward model
- verifier / step verifier
- reasoning supervision
- chain-of-thought 质量控制
- outcome reward vs process reward

的工作。

所以如果课程叫 **Process Reward in Large Language Models**，这篇几乎就是最核心的代表作之一。

---

## 5. 论文背景：它到底在回应什么问题

作者观察到一个现象：

虽然大型语言模型已经会写出“看起来像推理”的 chain-of-thought，但它们仍然常常在中间某一步胡说，最后甚至可能：

- 得到错误答案；
- 或者更糟，**用错误推理碰巧得到正确答案**。

这就带来一个问题：

如果你只根据最终答案来训练 reward model，那么 reward model 可能会把“错误但刚好答对”的解法也当成好解法。

于是 outcome supervision 会遇到两个困难：

1. **credit assignment 很差**
   它只知道“整条解答错了”，却不知道具体哪一步错了。
2. **会奖励伪推理**
   也就是 reasoning path 不对，但 final answer 偶然对。

而 process supervision 的优点是：

- 它能指出第一个错误出现在哪里；
- 它能给 reward model 更密、更精确的训练信号；
- 它更接近“我们真的想让模型学会什么样的推理过程”。

---

## 6. 关键术语，你在报告里最好先讲清楚

### 6.1 Outcome Supervision

只根据最终答案对不对来打标签。

你可以直接说：

**“Outcome supervision 只看终点，不看过程。”**

### 6.2 Process Supervision

对中间每一步推理都给反馈，至少要标出第一处错误出现在哪里。

你可以说：

**“Process supervision 不只是问你最后做对没做对，而是问你每一步是不是合理。”**

### 6.3 ORM

`Outcome-supervised Reward Model`

只学会对整段解答的最终质量打分。

### 6.4 PRM

`Process-supervised Reward Model`

对每一步打分，再综合成整条解答的分数。

### 6.5 Best-of-N Search

先让 generator 生成很多个解答候选，再让 reward model 从中挑最好的那个。

也就是说，这篇论文评估的重点不是“直接生成一个答案”，而是：

**“哪个 reward model 更会从很多候选解答里选出真正正确的那个。”**

---

## 7. 作者的方法到底做了什么

这部分是全篇最重要的“技术骨架”。

### 7.1 总体框架

作者固定一个 `generator` 来生成很多数学题解答，然后比较两种 reward model：

- ORM：只根据最终答案对错训练
- PRM：根据每一步的人工标签训练

最后评估时，让 reward model 在许多候选解答中选最优。

### 7.2 数据从哪里来

作者在 MATH 数据集上做实验。

对 process supervision，他们请人工标注者去看模型生成的逐步解答，并给每一步标：

- positive：这一步正确且合理
- negative：这一步错误或不合理
- neutral：有歧义，或者 technically 可以接受但表达不好

他们最终构建了 **PRM800K**：

- 约 **800K** step-level labels
- 来自约 **75K** solutions
- 覆盖约 **12K** problems

### 7.3 一个很重要的细节：只监督到第一处错误

作者特地选择：

**对错误解答，只标到第一处错误为止。**

这样做的好处是：

- 让 outcome supervision 和 process supervision 的比较更公平；
- 对错误解答，二者都知道“至少有错”；
- 而 process supervision 的额外信息只是“错在这里”。

你在 seminar 里可以强调：

**这其实是一个比较保守的设定。哪怕只比到这里，process supervision 还是更好。**

### 7.4 PRM 怎么给整条解答打分

PRM 会预测每一步“是正确的概率”。

然后作者把整条解答的分数定义成：

**所有步骤都正确的概率**

实现上，就是把每一步正确概率乘起来。

直觉上非常合理：

- 只要某一步很可疑，整条解答分数就会掉下来；
- 因为在多步数学推理里，一步错往往就足以让整题失效。

---

## 8. 实验设计怎么讲最清楚

### 8.1 Large-scale 实验

作者在大规模设置下，从 GPT-4 base model 微调出 reward models。

这里的目标是：

**尽量把 ORM 和 PRM 都做到“各自最强”，看谁更能筛选出正确解。**

评估方式是 `best-of-N`：

- 对每道题生成很多个候选解答；
- 让 reward model 排序；
- 取分最高的那个；
- 再看最终选出的解答是否正确。

### 8.2 Small-scale 实验

作者也做了一个小规模实验，原因是 large-scale 的训练数据分布不完全可比。

因为：

- PRM 的数据来自人工标注；
- 而且用了 active learning，数据更偏向“看起来像对但其实错”的解答；
- ORM 则可以直接靠 final answer 自动打标签。

为了更公平地比较两种 supervision，作者让一个更大的 PRM 去近似人类标注，再训练较小的 reward model，做受控 ablation。

这个设计很重要，因为它让论文不只是“秀结果”，而是在努力回答：

**process supervision 的优势，到底来自监督形式本身，还是来自别的实验因素？**

---

## 9. 最核心的结果

### 9.1 大规模结果：PRM 明显优于 ORM

论文最常被引用的结果之一是 Figure 3：

- ORM：**72.4%**
- PRM：**78.2%**
- Majority Voting：**69.6%**

这里是在代表性的 MATH 测试子集上，用大量候选解答做 `best-of-1860` 搜索得到的结果。

你可以直接口头总结成：

**“过程监督训练出来的 reward model，比结果监督更会从大量候选解答里挑出真正正确的解。”**

而且一个很漂亮的现象是：

**随着 N 增大，PRM 的优势还会变大。**

这说明 PRM 不只是“略强一点”，而是更适合在搜索空间里做高质量筛选。

### 9.2 小规模对照：优势不是偶然

在 small-scale synthetic supervision 里，作者尽量控制变量后，仍然观察到：

- process supervision 优于 outcome supervision；
- 即使 outcome supervision 不再只靠 final answer，而是用强 PRM 近似提供 outcome-style 监督，PRM 仍更强。

这说明：

**优势不只是因为 final-answer grading 太粗糙，而是 process supervision 本身就更有信息量。**

### 9.3 Active Learning 很有用

作者还发现：

**active learning 能带来大约 2.6 倍的数据效率提升。**

他们的想法是，不把人力浪费在“明显错得很离谱”的解答上，而是优先标那些：

- 最有迷惑性
- 最容易骗过当前 reward model
- 最值得人类去纠正

这对 seminar 很重要，因为它说明：

**process supervision 虽然贵，但不是完全不可扩展；通过挑选最有价值的数据去标，可以把成本压下来。**

### 9.4 OOD 泛化也更好

作者在新的 AP Calculus、AP Chemistry、AP Physics、AMC10/12 题目上也做了测试。

聚合结果是：

- ORM：**63.8%**
- PRM：**72.9%**
- Majority Voting：**61.3%**

这说明 PRM 的优势不只体现在 MATH 测试集上，对一定程度的 distribution shift 也有帮助。

---

## 10. 这篇论文真正有价值的解释

如果老师问你：

**“为什么 process supervision 会更强？”**

你最好的回答不是“因为它更细”，而是：

### 10.1 更好的 credit assignment

Outcome supervision 只告诉你“整题错了”。

但数学推理是长链条任务，一整条解答里可能前面很多步都对，只有某一步出错。

如果标签只有最终对错，那 reward model 学不到：

- 前面哪些步骤其实是好的；
- 真正的问题点在哪里。

而 process supervision 直接把这些信息给出来了。

### 10.2 更少奖励 hacking / spurious reasoning

如果一个错误推理碰巧得到正确答案，outcome supervision 容易把它当正样本。

PRM 则更容易看穿：

- 最终答案虽然对，
- 但中间 reasoning 不对，
- 所以这不是一条应该被奖励的思路。

### 10.3 更符合 alignment 直觉

作者专门讨论到 alignment impact：

- process supervision 更可解释；
- 人类更容易审核；
- 它奖励的是“人类认可的过程”，而不仅是“看起来好的结果”。

作者甚至把这个现象描述成一种 **negative alignment tax**：

也就是说，这里“更安全/更对齐”的做法，不但没有降性能，反而提高了性能。

这个点很适合拿来做结尾升华。

---

## 11. 你可以怎么批判这篇论文

一场好的 seminar 不能只讲优点，也要讲限制。

### 11.1 领域比较窄

这篇论文主要做的是数学推理。

数学有一个优势：

- final answer 通常比较容易自动检查。

但在开放域推理、法律、医学、复杂规划中，“正确步骤”怎么定义、怎么标，可能更难。

所以一个自然问题是：

**process supervision 在更开放、更模糊的任务上是否同样有效？**

### 11.2 标注成本很高

虽然 active learning 缓解了问题，但人工逐步标注仍然很贵。

所以它在工业上是不是能大规模推广，要看：

- 任务价值够不够高；
- 自动 verifier 能不能部分替代人类；
- 或者强模型能不能给弱模型提供 synthetic supervision。

### 11.3 PRM 自己也可能有偏差

PRM 的判断标准来自人类标注规则与数据分布。

如果标注者偏好某种“看起来规范”的写法，PRM 可能会偏向：

- 形式上更像标准答案的推理；
- 但未必总是最简洁或最创造性的推理。

### 11.4 评估还是基于最终答案

虽然 reward model 是 process-based 的，但最后评估“是否 solved”仍然看 final answer。

所以从严格意义上说，它最终还是在用 outcome metric 衡量 process method 的价值。

这没问题，但你可以指出：

**论文证明的是：process supervision 能更好地优化 outcome-defined success，而不是说它已经完整刻画了所有“好推理”。**

---

## 12. 适合你 presentation 的推荐结构

如果你是做一个大约 `20-30 分钟` 的 seminar，我建议下面这个结构。

### Slide 1. 标题页

- 论文标题
- 作者
- 课程名
- 你的名字

开场一句可以说：

**“今天我想讲的一篇非常经典的论文，是为什么在训练推理型语言模型时，我们应该奖励正确的过程，而不只是正确的结果。”**

### Slide 2. 问题背景

讲清楚：

- LLM 会 chain-of-thought
- 但中间步骤会错
- 最终答案对，不代表推理对

这一页的目标是制造 tension：

**“如果结果可以骗过我们，那奖励到底该给谁？”**

### Slide 3. 两种监督方式

做一个非常简单的对比图：

- outcome supervision：只看 final answer
- process supervision：逐步检查 reasoning

这一页可以是全场最关键的概念图。

### Slide 4. 论文核心问题

直接写：

**Process supervision 是否真的比 outcome supervision 更有效？**

然后补一句：

- 不只是更可解释
- 而是性能上也更好

### Slide 5. 方法概览

画出流程：

`Generator -> many candidate solutions -> Reward Model ranks -> pick best`

然后标清：

- ORM 怎么训练
- PRM 怎么训练

### Slide 6. PRM800K 和标注方式

讲：

- 每一步标 positive / negative / neutral
- 只标到第一处错误
- 80 万条 step-level labels

这页是为了让听众理解：process reward 不是空想，而是有真实数据流程支撑的。

### Slide 7. 主要结果 Figure 3

重点讲：

- PRM 78.2%
- ORM 72.4%
- Majority Voting 69.6%

这里别只报数字，要讲含义：

**“PRM 更会在大量候选中找到真正可靠的解答。”**

### Slide 8. 为什么会更好

这一页讲解释：

- 更好的 credit assignment
- 更少奖励错误但碰巧答对的 reasoning
- 监督信号更密

### Slide 9. Active Learning

讲：

- process supervision 很贵
- 但可以优先标“最有价值的错误”
- 数据效率提升约 2.6x

### Slide 10. OOD 与意义

讲 OOD 结果和 broader impact：

- PRM 在新题上也更好
- 更适合 alignment 视角

### Slide 11. 限制与思考

这里是你的批判性理解：

- 成本高
- 数学域偏强
- 如何扩展到开放任务仍未知

### Slide 12. 结论

建议用一句非常干净的话收束：

**“这篇论文最重要的贡献，是把 reward learning 的单位从‘整题结果’推进到了‘推理步骤’，并证明这种更细粒度的监督既更可靠，也更有潜力与 alignment 目标一致。”**

---

## 13. 你在讲的时候可以直接用的串联句

下面这些句子你几乎可以直接搬进 slides speaker notes。

### 13.1 从背景过渡到问题

“大型语言模型现在已经很会写 step-by-step reasoning，但问题是，写出来不等于推得对。尤其在数学里，只要中间一步错了，后面整条链条都可能失效。”

### 13.2 从问题过渡到监督方式

“那接下来的问题就是，我们到底应该怎样奖励模型？是只看最后答案，还是检查它中间每一步到底是否合理？”

### 13.3 从监督方式过渡到方法

“这篇论文的做法非常直接：固定一个 generator 先生成很多候选解答，然后分别训练 outcome-supervised 和 process-supervised reward model，看谁更会挑出真正正确的解。”

### 13.4 从方法过渡到结果

“方法本身并不复杂，但真正关键的是，这种更细粒度的监督到底有没有实际收益。作者接下来用大规模和小规模两组实验回答这个问题。”

### 13.5 从结果过渡到解释

“看到 PRM 明显更强之后，下一个自然问题不是‘数字为什么更高’，而是‘它到底为什么更高’。论文给出的核心解释就是更好的 credit assignment。”

### 13.6 从解释过渡到意义

“所以这篇论文的重要性不只是把一个 benchmark 做高了，而是说明：如果我们想要更可靠的 reasoning model，奖励信号本身也要进入 reasoning process。”

### 13.7 结尾收束

“如果只奖励结果，模型可能学会的是如何显得正确；如果奖励过程，模型更有机会学会如何真正地正确。这就是这篇论文最经典的地方。”

---

## 14. 如果老师问你“这和 Process Reward 课程的关系是什么”

你可以这样答：

这篇论文几乎就是 `process reward` 思路的代表性工作之一。它把奖励建模从最终 outcome 推进到 step-level reasoning process，并实证说明：

- reward 可以作用在过程上；
- 过程奖励不只是更可解释，而且可能更强；
- process reward model 可以成为后续 verifier、reasoning alignment、test-time search 的基础模块。

所以它不是一篇普通的 math reasoning paper，而是一篇：

**“process reward 如何落地到 LLM reasoning” 的标志性论文。**

---

## 15. 可能会被问到的问题和简短答法

### Q1. 为什么 outcome supervision 不够？

因为它只告诉模型“整题对/错”，却不告诉模型“哪一步出错”。对于长链推理，这种监督太稀疏。

### Q2. 为什么 process supervision 更强？

因为它提供更细粒度的 credit assignment，能更准确地区分“前面是对的、后面才错了”的解答。

### Q3. 这是不是只适用于数学？

目前证据主要来自数学。数学比较适合做这类实验，但论文也明确认为，是否能推广到更开放领域还需要后续研究。

### Q4. process supervision 会不会太贵？

会，所以作者引入 active learning，并显示出大约 2.6x 的数据效率提升。这说明成本高，但并非不可管理。

### Q5. 这和 alignment 有什么关系？

因为 process supervision 奖励的是“人类认可的推理过程”，而不是只奖励最终表面上看起来好的结果，所以它更可解释，也更不容易鼓励 reward hacking。

---

## 16. 最后给你的一个讲法建议

如果你想把这篇讲得成熟一点，不要把它讲成：

`PRM > ORM，所以 process supervision 更强。`

而要讲成：

**“在长链推理任务里，监督信号本身的粒度决定了 reward model 能学到什么。Let's Verify Step by Step 的核心贡献，是证明 step-level supervision 既能提升搜索质量，也更接近我们真正想对齐的推理行为。”**

这样讲，会更像一场 seminar，而不是一篇论文摘要。

---

## 17. 你做 slides 时最值得保留的三句话

如果你最后只想记住三句话，我建议保留这三句：

1. **Outcome supervision 只看结果，process supervision 直接检查推理步骤。**
2. **在数学推理上，process-supervised reward model 比 outcome-supervised reward model 更可靠、更会选对答案。**
3. **这篇论文的重要意义，不只是提升性能，而是把 reward learning 的重点从“结果正确”推进到“过程正确”。**

---

## 18. 参考信息（便于你在 slides 末页引用）

- Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., & Cobbe, K. (2023). *Let's Verify Step by Step*. OpenAI / arXiv:2305.20050.
- 论文主结论包括：
  - PRM 在代表性 MATH 测试子集上达到 `78.2%`
  - ORM 为 `72.4%`
  - Majority voting 为 `69.6%`
  - active learning 带来约 `2.6x` 数据效率提升
  - 在 OOD STEM 测试上 PRM 聚合结果为 `72.9%`，优于 ORM 的 `63.8%`

---

## 19. 一句话总结版

**这篇论文证明：对于需要长链推理的 LLM，仅奖励最终答案是不够的；如果我们逐步验证并奖励中间推理过程，就能训练出更可靠的 reward model，这既提升性能，也更贴近 alignment 的目标。**
