![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-04/9e605a42-a07c-4090-a65a-d20b61dac81f/403893b432c6c81f0f14532bcbdab82e329367f5bac619f802cc9c0960656710.jpg)


# Agent-R1: A Unified and Modular Framework for Agentic Reinforcement Learning

Mingyue Cheng, Shuo Yu, Daoyu Wang, Qingchuan Li, Xiaoyu Tao, Jie Ouyang, Yucong Luo, Yitong Zhou, Qi Liu, Enhong Chen 

State Key Laboratory of Cognitive Intelligence, University of Science and Technology of China 

## Abstract

Large language models (LLMs) have rapidly evolved from single-turn text generators into the foundation of increasingly capable agents. As these agents take on more complex reasoning, decision making, tool use, and long-horizon tasks, reinforcement learning (RL) is becoming increasingly important for shaping their behavior. This shift is especially visible in agentic RL, where models must interact with tools and environments across multiple rounds rather than produce a single standalone response. In this regime, the usual view of a trajectory as one ever-growing token sequence becomes increasingly inadequate: it makes context evolution rigid and creates representation mismatches between rollout and training. This paper presents Agent-R1, a unified and modular framework for agentic RL built around step-level trajectory representation, flexible context management, and layered interfaces for workflows, environments and optimization. The key idea is to treat each interaction step as the basic reinforcement-learning transition, while keeping the optimization layer flexible: once the interaction is modeled at the step level, the framework can support token-level credit assignment, step-level credit assignment, or other compatible designs. These design choices make the framework compatible with a range of optimization strategies rather than tying it to a single algorithm. Together, these components provide a principled, extensible, and reusable substrate for agentic RL. 

Correspondence: mycheng@ustc.edu.cn Agent-R1: https://github.com/AgentR1/Agent-R1 

## 1 Introduction

Large language models (LLMs) [1, 4] were initially developed as single-turn text generators, producing one response for one prompt. In this setting, they are mainly treated as conditional sequence models for tasks such as text continuation, question answering, summarization, and instruction following [14]. As model capabilities improved, however, researchers began to build more complex systems around them. Through prompting, tool use, memory mechanisms, retrieval augmentation [39], and environment feedback, LLMs evolved from standalone generators into the foundation of increasingly capable agents [5]. This shift has enabled systems that can search, plan, use tools [22, 31, 37], and interact with environments over long horizons [28, 36]. As these systems take on more ambitious goals, the challenge is no longer only to improve a single response, but to optimize the multi-turn decision process that drives sustained agent behavior 

Reinforcement learning (RL) is therefore becoming an increasingly important post-training paradigm for 


Table 1 Representative open-source frameworks for agentic RL. Agent-R1 combines step-level MDP abstraction with flexible context management.


<table><tr><td>Framework</td><td>MDP abstraction</td><td>Context management</td></tr><tr><td>veRL [26]</td><td>Token-level</td><td>No context management</td></tr><tr><td>slime [29]</td><td>Token-level</td><td>No context management</td></tr><tr><td>Agent Lightning [16]</td><td>Step-level</td><td>Implicit context management</td></tr><tr><td>AReaL [8]</td><td>Not explicit</td><td>Implicit context management</td></tr><tr><td>rLLM [21]</td><td>Not explicit</td><td>Implicit context management</td></tr><tr><td>Agent-R1 [2]</td><td>Step-level</td><td>Flexible context management</td></tr></table>

LLMs [19, 25, 32, 40]. Earlier recipes such as Reinforcement Learning from Human Feedback (RLHF) [19] and Reinforcement Learning with Verifiable Rewards (RLVR) [25] were largely developed around single responses or short reasoning traces. In those settings, token-level generation remains a natural abstraction. Agentic RL introduces a diferent setting: the model must act across multiple rounds, interact with tools and environments, and adapt to external feedback over time. As a result, delayed and sparse rewards, long-horizon interaction, and long context all become first-class challenges. At the same time, the trajectory is no longer naturally viewed as a single growing response. Representing it as one ever-growing token sequence therefore becomes increasingly inadequate: it makes context evolution rigid and creates representation mismatches between rollout and training. 

In this paper, Agent-R1 starts from a step-level MDP abstraction, in which each interaction round becomes a proper RL transition with an observation, an LLM action, and an environment update. This modeling choice makes the interaction step, rather than the token, the native unit for organizing agent behavior. Early agentic RL frameworks often stored trajectories as messages and later reconstructed them as text for training [11]. While convenient, this creates a mismatch: rollout happens in token space, but training may rely on re-tokenized text. Since the token-text-token mapping is not reversible, this can introduce retokenization drift and break rollout-training consistency. Agent-R1 addresses this problem through a step-level trajectory representation in which each round preserves its observation and action boundary. It also addresses the rigidity of append-only context growth by allowing the environment to construct the next observation flexibly, rather than treating history as a flat sequence. 

Table 1 highlights the main distinction from representative agentic RL frameworks. Existing systems often make only one of these two design choices explicit: some retain token-level training abstractions, while others move closer to step-level interaction but leave context handling implicit. Agent-R1 emphasizes their combination within the same training substrate, making both the interaction unit and the context-construction rule explicit. This design not only preserves rollout structure while supporting task-dependent memory policies during training, but also lays a reusable foundation for future step-level credit-assignment research in the broader agentic RL community. 

## 2 Preliminaries

This section provides the background needed to understand why Agent-R1 is designed the way it is. We first review the current infrastructure landscape for LLM training and serving, then move from LLM training to LLM RL training, then from single-turn LLM RL to agentic RL, and finally summarize the concrete problems that arise once RL targets multi-turn agent behavior rather than single-turn text generation. 

## 2.1 Infrastructure for LLM Inference and Training

The recent ecosystem of LLM infrastructure is already highly modular. At the inference and serving layer, engines such as vLLM [12] and SGLang [24] provide eficient generation, batched decoding, and structured execution for LLMs. At the large-scale training layer, frameworks such as DeepSpeed [17], PyTorch Fully Sharded Data Parallel (FSDP) [41], and Megatron-LM [27] support distributed optimization and parallel training at scale. Together, these systems have established a standard separation of concerns: high-throughput inference and large-scale optimization are often implemented as distinct layers rather than one monolithic stack. This modular infrastructure is highly efective for conventional LLM development, but RL training must connect these two sides into one loop. Rollout depends on eficient inference engines, while optimization depends on scalable training frameworks. Once a model is embedded into an interactive workflow, the RL system must bridge them while also supporting tools, environment feedback, trajectory construction, and replay across many rounds. Like other agentic RL training frameworks, Agent-R1 is also designed to connect the inference side and the training side into a unified training loop. Agent-R1 is motivated by the observation that existing LLM infrastructure is necessary but not suficient for this setting. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-04/9e605a42-a07c-4090-a65a-d20b61dac81f/b7db220ae97d807390abdc1ba1345223775b7adc72f6dbe173928b66c0cacf01.jpg)



Figure 1 Rollout in agentic RL. The agent repeatedly observes the current context, produces actions, and receives feedback from tools or environments, forming a multi-step trajectory rather than a single prompt-response pair.


## 2.2 From Supervised LLM Training to RL Training

LLM RL training difers from standard supervised LLM training mainly in its execution loop. In standard supervised training, data are prepared in advance, batches are fed into the model, and optimization proceeds directly on the resulting losses. In LLM RL training, by contrast, optimization depends on rollout: the model must first generate responses, those responses must then be evaluated by rewards or preferences, and the resulting trajectories are sent back to the optimizer. This introduces a new dependency between inference and training that is largely absent from ordinary supervised learning. Typical examples include RLHF [19] and RLVR [25], optimized with methods such as PPO [23] and GRPO [25]. Compared with standard LLM training, the pipeline must now support sampling, reward computation, and replay in addition to optimization itself. As a result, LLM RL training is not just a new loss function layered on top of LLM training; it is a diferent end-to-end loop that must coordinate rollout and optimization. 

## 2.3 From Single-Turn LLM RL to Agentic RL

Agentic RL extends this change one step further. In single-turn or short-horizon LLM RL, rollout still mostly takes the form of one prompt and one response. In agentic RL, rollout becomes a multi-round interaction among the model, its tools, and the environment [22, 28, 32, 36, 37]. As illustrated in Figures 1 and 2, the framework must therefore connect rollout with trajectory replay and policy optimization, while handling environment feedback, multi-step trajectory construction, and interactive traces [8, 10, 16]. This shift changes what it means to preserve rollout faithfully. Rollout and optimization can no longer be treated as the same prompt-response abstraction; the framework must keep the causal structure of observation, action, feedback, and termination across multiple steps [10]. This requirement brings three practical consequences. First, trajectories need an explicit step-level representation rather than a flat generation sequence. Second, context construction must remain flexible, since diferent tasks may require selecting, reconstructing, or compressing prior interaction in diferent ways [32, 37]. Third, the system must connect LLM inference, tool execution, environment simulation, and optimization even though these components often operate at diferent timing and compute granularities [8, 15, 16, 21, 28, 36]. These issues motivate the step-level formulation introduced next. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-04/9e605a42-a07c-4090-a65a-d20b61dac81f/1f0c1ba9d0f11457639b2d67807415f28d2da18d5f0ec30f09f50d1f26a8ad7a.jpg)



Figure 2 Optimization in agentic RL. Interactive trajectories collected during rollout are replayed by the training loop and used to update the policy through reinforcement-learning objectives.


## 2.4 From LLMs to Agents: An MDP Perspective

The previous subsection described the practical shift from single-turn generation to multi-turn interaction in RL. The same shift can be formalized through the lens of Markov Decision Processes (MDPs). In the static single-turn setting, the state is typically the current textual context: 

$$
s _ {t} = (\mathbf {w} _ {p}, w _ {1}, w _ {2}, \dots , w _ {t}),\tag{1}
$$

where $\mathbf { w } _ { p }$ is the initial prompt and $w _ { 1 } , \ldots , w _ { t }$ are the generated tokens. The action $a _ { t }$ is the next token, and the transition is deterministic: 

$$
P (s _ {t + 1} \mid s _ {t}, a _ {t}) = \left\{ \begin{array}{l l} 1, & \text { if } s _ {t + 1} = s _ {t} \oplus a _ {t}, \\ 0, & \text { otherwise }. \end{array} \right.\tag{2}
$$

This formulation is adequate for single-turn generation, but becomes restrictive once the model must interact with environments across multiple rounds. 

For an LLM agent, the observable context must also capture prior interaction turns and environmental feedback. We therefore use an observation-centric step-level formulation: 

$$
o _ {t} = (\mathbf {w} _ {p}, \mathcal {T} _ {1}, \mathcal {T} _ {2}, \ldots , \mathcal {T} _ {k}, \mathcal {T} _ {k + 1} ^ {\mathrm{partial}}),\tag{3}
$$

where each $\mathcal { T } _ { i }$ denotes a complete interaction turn and $\mathcal { T } _ { k + 1 } ^ { \mathrm { p a r t i a l } }$ the ongoing partial turn. Under this view, Agent-R1 treats each interaction round, rather than each token, as the native action unit for training and optimization. Accordingly, a rollout is written as a sequence of structured step traces: 

$$
\tau = \{z _ {t} \} _ {t = 0} ^ {T - 1}, \qquad z _ {t} = (o _ {t}, a _ {t}, e _ {t}, r _ {t}, o _ {t + 1}),\tag{4}
$$

where $a _ { t }$ may be a natural-language response, a structured tool invocation, or a mixed output containing both reasoning and external actions, and $e _ { t }$ denotes the environment feedback returned after executing $a _ { t }$ . 

At the interface level, the environment defines a step transition operator: 

$$
\mathcal {E} (o _ {t}, a _ {t}) = (o _ {t + 1}, r _ {t}, d _ {t}, e _ {t}),\tag{5}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-04/9e605a42-a07c-4090-a65a-d20b61dac81f/912dd1a0c0111745a43e69d361a4d975f02de99d5af0b2ffc70f4bd312d0e886.jpg)



Figure 3 Illustration of an Agent-R1 training trajectory. The left side shows step-level trajectory representation, where each interaction step is recorded as a structured unit rather than flattened into one long sequence. The right side shows flexible context management, where environment feedback can be appended, summarized, or otherwise reorganized before constructing the next-step context.


where $d _ { t }$ denotes termination and $e _ { t }$ the environment feedback returned at step t. This step-level formulation naturally accommodates both terminal outcome rewards and intermediate process feedback, which is a key distinction between agentic RL and ordinary single-turn RL. 

The same formalization also clarifies flexible context management. Rather than requiring the next observation to arise from simple append-only concatenation, Agent-R1 allows the environment to construct it explicitly: 

$$
o _ {t + 1} = \mathcal {C} (z _ {0}, z _ {1}, \dots , z _ {t}),\tag{6}
$$

where $\mathcal { C }$ is the context-construction rule applied to the structured interaction history. The resulting context may preserve or transform prior interaction history while remaining within a well-defined transition. This is why a step-level MDP provides a more natural foundation for multi-turn agent training. 

## 3 Agent-R1

## 3.1 Overview of Agent-R1

The role of Agent-R1 is to connect and unify two sides that are often developed separately in practice. On one side are agentic RL algorithms, including optimization objectives, reward definitions, advantage estimation, and credit assignment. On the other side are infrastructure concerns, including workflow execution, rollout sampling, model serving, and large-scale optimization. Agent-R1 serves as the bridge between these two sides, so that algorithmic ideas can be instantiated on top of a common training substrate rather than being reimplemented together with the full execution stack each time. 

Agent-R1 adopts interaction steps as the native training unit. Each step preserves its observation, action, feedback, and next observation, making step-level trajectory representation and flexible context management native to the training loop. At the interface level, the environment can be viewed as a step transition operator: 

$$
\mathcal {E} (o _ {t}, a _ {t}) = (o _ {t + 1}, r _ {t}, d _ {t}, e _ {t}),\tag{7}
$$

where $o _ { t }$ is the current observation, $a _ { t }$ is the agent action for the current step, $o _ { t + 1 }$ is the next observation, $r _ { t }$ is the reward, $d _ { t }$ is the termination flag, and $e _ { t }$ denotes the environment feedback returned during the transition. Agent-R1 is designed so that workflows, tools, and optimization modules all communicate through this step-native interaction boundary. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-04/9e605a42-a07c-4090-a65a-d20b61dac81f/42538e71cf0c5cae99d6c83ed7227b01b5d7ba78a9846529b587465fe44d0aa0.jpg)



Figure 4 Trajectory representation evolves from message-based traces to token-consistent records and finally to structured step-level traces. This progression is important not only for replay correctness, but also for flexible context management in multi-turn agent training.


To support optimization, rollout must preserve enough structure to distinguish complete actions, intermediate feedback, and final outcomes. A rollout can therefore be written as a sequence of step traces: 

$$
\tau = \{z _ {t} \} _ {t = 0} ^ {T - 1}, \qquad z _ {t} = (o _ {t}, a _ {t}, e _ {t}, r _ {t}, o _ {t + 1}),\tag{8}
$$

where $e _ { t }$ denotes the environment-side feedback generated after executing action $a _ { t }$ . This representation allows the learning side to distinguish complete agent actions from environment, attach both outcome rewards and process rewards to the appropriate steps, and keep replay faithful to the interaction trajectory. 

For optimization, one useful abstraction is the action mask over generated tokens within each step. If the token sequence emitted at step t is written as $a _ { t } = ( y _ { t , 1 } , \dots , y _ { t , L _ { t } } )$ , then a mask is defined as: 

$$
m _ {t, j} = \left\{ \begin{array}{l l} 1, & \text { if   } y _ {t, j} \text {   belongs   to   the   agent   action   at   step   } t, \\ 0, & \text { otherwise } \end{array} \right.\tag{9}
$$

This mask selects the tokens that should receive policy-gradient updates, while leaving prompt tokens and environment-side content outside the policy loss. A generic masked policy objective can then be written as: 

$$
\mathcal {L} _ {\text { policy }} = - \sum_ {t = 0} ^ {T - 1} \sum_ {j = 1} ^ {L _ {t}} m _ {t, j} \hat {A} _ {t, j} \log \pi_ {\theta} (y _ {t, j} \mid o _ {t}, y _ {t, <   j}),\tag{10}
$$

where the credit term $\hat { A } _ { t , j }$ may be instantiated either at token level or by broadcasting a step-level signal across the action tokens of step t. In this way, Agent-R1 standardizes the trajectory substrate used by optimization while remaining compatible with PPO-style, GRPO-style, and other RL objectives. 

## 3.2 Step-level Trajectory Representation

A central design choice in Agent-R1 is how multi-turn trajectories are represented for replay and optimization. In existing agent training pipelines, trajectory representations often fall into two common forms: message traces and flat token sequences. Message traces are convenient for workflow construction and debugging, but they do not preserve the exact token sequence generated during rollout. If Tok(·) denotes tokenization and Text(·) denotes text reconstruction, the replayed sequence is typically: 

$$
\tilde {x} _ {1: N ^ {\prime}} = \operatorname{Tok} \bigl (\operatorname{Text} (\mathcal {M} (\tau)) \bigr),\tag{11}
$$

whereas the original rollout is generated on: 

$$
x _ {1: N} = (x _ {1}, x _ {2}, \ldots , x _ {N}).\tag{12}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-04/9e605a42-a07c-4090-a65a-d20b61dac81f/9251757c3e45c764af107e1906f810939ac1f5d787f9db428540a6e9523f79f3.jpg)



Figure 5 A multi-hop question-answering example of flexible context management in Agent-R1. The same step-level replay record can support multiple environment-defined next-observation constructions, including append-only history growth, evidence-focused selection, and summarized task state.


As discussed earlier, these two sequences need not be identical [10]. Such mismatch may shift action boundaries, alter the efective action mask, and distort the log-probabilities used during optimization. Flat token-space storage avoids this retokenization problem by preserving the exact rollout tokens, but it still treats the interaction as one append-only sequence and leaves step boundaries implicit. 

Agent-R1 addresses these limitations by adopting step-level traces as the native trajectory abstraction. As introduced above, each rollout is stored as a sequence of structured step records, so that replay remains faithful to the original interaction while the step boundary stays explicit. By making the step boundary explicit, Agent-R1 can identify not only which tokens were generated, but also which token span corresponds to one complete agent action, what feedback it triggered, and how that feedback changed the next observation. If the action at step t consists of tokens $( y _ { t , 1 } , \dots , y _ { t , L _ { t } } )$ , then rewards, masks, and credit signals can be aligned either with these internal action tokens or with the step $z _ { t }$ as a whole. 

The practical benefit is that step-level traces provide a common substrate for multiple optimization views. Token-level objectives can still be applied over the action tokens within each step, while step-level rewards or process supervision can be attached directly to the corresponding transition. In this way, step-level replay is not a derived convenience, but the native representation used by Agent-R1 for multi-turn agent RL. 

## 3.3 Flexible Context Management

Flexible context management is the second key design choice in Agent-R1. As illustrated in Figure 5, the next context in a multi-turn agent system can be constructed in multiple ways. This is important because tool outputs may be verbose, intermediate reasoning may be irrelevant, and long interaction histories may exceed the useful context budget for the next decision. Rather than hard-wiring a fixed append-only strategy, Agent-R1 allows the next observation to be defined by an environment-specific context rule: 

$$
o _ {t + 1} = \mathcal {C} (z _ {0}, z _ {1}, \dots , z _ {t}),\tag{13}
$$

where the visible context is constructed from structured interaction history rather than blind concatenation. This allows prior traces to be preserved, summarized, omitted, or otherwise transformed while remaining in the replay record, and naturally supports more general memory-management strategies 

This flexibility is one of the key reasons step-level representation matters. Once trajectories are stored as structured steps rather than one flat text stream, context construction no longer needs to be identical to raw replay. Agent-R1 can keep an exact replayable record for optimization while allowing the environment to decide what should be exposed to the model at the next step. In this sense, trajectory representation and context management are not independent features, but two sides of the same step-level design choice. 


Table 2 Main experimental results across representative application scenarios under Agent-R1. The best result in each column is in bold, and the second-best result is underlined.


<table><tr><td rowspan="2">Method</td><td rowspan="2">GSM8K Acc. (%)</td><td rowspan="2">HotpotQA Acc. (%)</td><td colspan="2">ALFWorld</td><td colspan="2">WebShop</td></tr><tr><td>SR (%, Seen)</td><td>SR (%, Unseen)</td><td>Score (%)</td><td>SR (%)</td></tr><tr><td>ReAct</td><td>53.1</td><td>25.8</td><td>7.14</td><td>2.98</td><td>51.58</td><td>23.8</td></tr><tr><td>GRPO</td><td>83.3</td><td>59.4</td><td>81.29</td><td>74.58</td><td>65.83</td><td>44.2</td></tr><tr><td>PPO</td><td>78.1</td><td>56.7</td><td>76.42</td><td>72.38</td><td>70.18</td><td>46.0</td></tr><tr><td>Reinforce++</td><td>78.9</td><td>52.8</td><td>73.84</td><td>69.57</td><td>63.41</td><td>41.8</td></tr><tr><td>RLOO</td><td>81.6</td><td>55.2</td><td>79.08</td><td>73.46</td><td>68.02</td><td>45.1</td></tr></table>

## 4 Experiments

We evaluate two questions: whether Agent-R1 transfers across diferent agent tasks, and whether its contextmanagement interface afects learning quality under a fixed training setup. 

## 4.1 Experimental Setting

We instantiate Agent-R1 with Qwen3-4B[34] on GSM8K[6], HotpotQA[35], ALFWorld[28], and WebShop [36]. These tasks span arithmetic reasoning with sandboxed coding, retrieval-based multi-hop question answering, embodied household interaction, and simulated online shopping. The controlled comparisons below focus on GSM8K, where the Agent-R1 environment, tool-based interaction setting, tool format, and reward definition are fixed, so that diferences can be attributed more directly to the optimization algorithm or context-management rule. The reward combines answer accuracy with a format component. 

## 4.2 Different Application Scenarios

Table 2 summarizes the main results across representative scenarios. We report one representative task metric for each setting and compare GRPO, PPO, Reinforce++, and RLOO under the same Agent-R1 framework. All four RL methods outperform the training-free baseline across these diverse settings. At the same time, the best optimizer varies by task: GRPO leads on the arithmetic, retrieval, and embodied settings, while PPO is strongest on the shopping task. This pattern suggests that Agent-R1 is broad enough to support heterogeneous agent environments while still preserving meaningful algorithmic diferences. Figure 6 shows representative training curves on three distinct tasks under GRPO. All three exhibit clear upward trends, indicating that Agent-R1 can support efective learning across heterogeneous agent settings. At the same time, their optimization dynamics difer substantially: GSM8K improves rapidly and stabilizes early at a high level, HotpotQA shows slower and more fluctuating gains, and ALFWorld exhibits a more stage-wise improvement pattern with pronounced late-stage jumps. This suggests that while the same framework transfers across tasks, the underlying learning dynamics remain task-dependent. 

## 4.3 Different RL Algorithms

We further compare PPO, GRPO, Reinforce++, and RLOO under the same GSM8K environment to isolate the efect of the optimizer. Figure 7 reports reward, accuracy, and response length under matched prompts, tool format, and rollout configuration. Two patterns are especially notable. First, GRPO and RLOO reach the strongest late-stage accuracy, while PPO remains more volatile. Second, Reinforce++ behaves diferently from the other optimizers: although it still achieves relatively high accuracy, its reward remains substantially lower. This discrepancy is consistent with its much shorter responses in the later stage of training. Since the reward in this setting combines answer accuracy with a format-related component, Reinforce++ appears to learn a more conservative policy that can still produce correct answers, but is less efective at maximizing the full training signal. This highlights that, in multi-turn tool-augmented RL, high task accuracy does not necessarily imply high reward, and diferent optimizers may favor diferent response strategies. This shows that Agent-R1 does not wash out optimizer-specific behavior; it makes that behavior observable under a common interaction setup. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-04/9e605a42-a07c-4090-a65a-d20b61dac81f/7bd0bb9cce6a78ea9e8ff5c9fe3d635b858dd8508c64226872dc61600eeb2e02.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-04/9e605a42-a07c-4090-a65a-d20b61dac81f/97752a268856781a0dc51732fbe5f8d75fb72cfb82e0567a0db930043807fc7a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-04/9e605a42-a07c-4090-a65a-d20b61dac81f/04c28f32005b6cd54c4b5fab9082a04f4f125803698fcdba1735ee444a1d6ccc.jpg)



Figure 6 Representative training curves of Agent-R1 across multiple application scenarios. We show results on GSM8K, HotpotQA, and ALFWorld as three representative examples to illustrate that the same framework can be instantiated for tool-augmented mathematical reasoning, multi-hop question answering, and interactive decision-making tasks.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-04/9e605a42-a07c-4090-a65a-d20b61dac81f/d5b0d921b560018f796dd9da7e5ae65b8c7068ab1e40e8ed12ec080e5e5fbbf8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-04/9e605a42-a07c-4090-a65a-d20b61dac81f/aece29ecc546ba337220b1d6e791ae54ead443d7be5c2a51dad9b584cc8042d4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-04/9e605a42-a07c-4090-a65a-d20b61dac81f/4540d3ba8849d734caf5b31fc2bde25bd975c97af1c78a55689357cb7a65378f.jpg)



Figure 7 Training curves on GSM8K under Agent-R1 with diferent RL algorithms. We compare PPO, GRPO, Reinforce++, and RLOO under the same environment, tool format, and rollout configuration, and report reward, accuracy, and response length to show how optimizer choice afects both task performance and response behavior.


## 4.4 Context-Management Strategies

To test whether flexible context construction matters in practice, we compare append-only, sliding-window, and LLM-summarized context under the same GRPO setup on GSM8K. Figure 8 shows that sliding-window context performs best, while direct append-only replay is weaker and summary-based context underperforms in this small-model setting. The result supports the main design claim of Agent-R1: context management is not just a presentation detail. When the framework exposes context construction explicitly, it becomes possible to study how diferent memory rules afect training under the same rollout and optimizer. In this experiment, preserving only the most relevant recent evidence produces a cleaner learning signal than either unbounded history growth or noisy model-generated summaries. The poor performance of LLM-summarized context in this setting should not be read as a general rejection of summary-based memory; it instead suggests that the quality of the transformation itself becomes part of the training problem. The response-length curves are broadly consistent with this interpretation: sliding-window context keeps responses more controlled without hurting task performance, whereas the other two strategies either retain excessive history or introduce lossy compression. 

## 5 Future Directions

Agent-R1 is designed as a starting point rather than an endpoint, and several future directions arise from the step-level view itself. A first question is what the next generation of trajectory representation should look like. The current step-level trace already makes context management more flexible and creates a cleaner substrate for step-level credit assignment, but it can still introduce computational redundancy during training because many trajectories share long common prefixes while being optimized separately. Recent systems such as MiniMax Forge have begun to explore prefix-sharing and tree-structured merging to reduce repeated computation across related trajectories [18]. A natural next step is therefore to ask how structured step traces can preserve replay fidelity and context flexibility while also exposing more eficient computation graphs for optimization. A second direction concerns how increasingly complex agent environments should be connected to the training framework. As agents move into settings with richer tools, branching workflows, persistent state, and delayed feedback, the framework should make environment integration as non-intrusive as possible without drifting too far from the on-policy interaction that the optimizer assumes [8, 16]. Balancing these two goals is dificult: tighter integration often improves faithfulness, while looser integration often improves usability. A third direction is how to obtain higher-quality RL data for agent training. In multi-turn settings, data quality is determined not only by final success or failure, but also by whether trajectories contain informative intermediate decisions, meaningful tool interactions, and useful exploration patterns [11, 32]. This suggests that future progress may depend as much on better data generation, filtering, and curriculum design as on the optimizer itself. In this sense, the long-term challenge is not only to make agent training easier to run, but to identify the right data and abstraction for learning robust multi-turn agent behavior. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-04/9e605a42-a07c-4090-a65a-d20b61dac81f/10f46f21755c830432cc82de681e1d5bd7593ee6e26a5403d899d8e7f52ed408.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-04/9e605a42-a07c-4090-a65a-d20b61dac81f/60cdb0e7bd06756cdbf7a8cd2de3d4c94440eb4c23bfa1ca8a4935b340ed297a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-04/9e605a42-a07c-4090-a65a-d20b61dac81f/6901c633a501f736f48aa425a3c3c3b35f0a09dd2156596f409cba1250c50e9d.jpg)



Figure 8 Comparison of three context-management strategies on GSM8K under the same GRPO training setup. We compare append-only context, sliding-window context that keeps the original question together with the most recen tool output and model analysis, and LLM-summarized context that compresses the evolving interaction history.


## 6 Limitation Discussion

Although Agent-R1 provides a unified and modular framework for agentic reinforcement learning, it still has several limitations. First, the step-level trajectory representation may bring extra training cost. When an agent uses append-only history without explicit context management, diferent training samples may share long prefixes. Since these step-level records are trained separately, the same prefix can be computed many times, causing redundant computation. This issue becomes more serious for long-horizon agents with many interaction steps. Future work may reduce this cost through prefix sharing, KV-cache reuse, or tree-structured trajectory merging, while keeping the replay fidelity and context flexibility of step-level traces. 

Second, Agent-R1 has not fully explored asynchronous rollout-training execution. Although rollout workers and the training backend are decoupled, the current system still requires collected trajectories to be aligned with optimization updates. This design is practical for simple or moderately long tasks, but it can be ineficient for complex agents with slow tools, branching workflows, persistent environments, or delayed feedback. In these cases, rollout may become the main bottleneck, making training less eficient than fully asynchronous systems. Improving asynchronous scheduling and scalable rollout-training coordination is an important direction for future work. 

## 7 Related Work

This section situates Agent-R1 within the emerging literature on agentic RL, with emphasis on two closely related questions: how multi-turn agent behavior is optimized, and how training frameworks organize that optimization in practice across diferent agent settings. 

## 7.1 Agentic RL Algorithms

Agentic RL algorithms inherit their basic optimization toolbox from earlier RL post-training work such as PPO [23] and GRPO-style training [25], but they must adapt these ideas to multi-turn interaction, sparse feedback, and environment-coupled behavior. Early work such as Search-R1 [11] and RAGEN [32] already made these dificulties visible by showing that response-level rewards are often too weak to supervise long interaction chains. A natural trend that followed was to move from flat sequence optimization toward turn-aware and step-aware credit assignment. This transition is also related to broader sequence-level directions such as DAPO [38] and GSPO [42], which make the optimization unit less token-local even outside fully agentic settings. Many later methods can then be understood as extensions of the PPO and GRPO families in a more explicitly agentic direction. On the PPO side, Turn-PPO [13] reformulates multi-turn training around turn-level advantage estimation, StepPO [30] pushes this idea further by aligning credit assignment with agent steps rather than individual tokens, and PaperScout [20] emphasizes process-aware sequence-level optimization for literature-search agents. On the GRPO side, Tree Search for LLM Agent Reinforcement Learning [9] and GiGPO [7] both move beyond flat response-level grouping and attempt to expose finer-grained structure inside multi-step agent rollouts. In parallel, another line of work focuses less on changing the base optimizer itself and more on strengthening the supervision signal around it. AgentPRM [33] studies step-wise process reward modeling for agent trajectories, and SWEET-RL [43] explores the use of privileged training-time information to improve supervision over multi-turn reasoning behavior. Taken together, these methods suggest a clear trend: agentic RL is moving away from treating the entire rollout as a single undiferentiated response, and toward optimization schemes that respect the internal structure of interaction. 

## 7.2 Agentic RL Training Frameworks

In parallel with algorithmic advances, agentic RL training frameworks have also become a distinct and rapidly evolving research topic. One line of development starts from general RL post-training substrates and gradually extends them toward agent workloads. veRL [26] provides a strong foundation for distributed RL post-training, while slime [29] emphasizes high-performance scaling and flexible generation workflows. A second line focuses on how to connect already-built agents to RL with minimal friction. Agent Lightning [16] foregrounds decoupled agent-training architecture and broad compatibility with existing agents, while rLLM [21] emphasizes low-intrusion integration with external agent frameworks. A third line places more weight on execution topology and system scalability. AReaL [8] focuses on asynchronous rollout-training execution, MiniMax Forge [18] highlights eficient large-scale agent RL workloads, and Claw-R1 [3] extends this design space toward a more deployment-oriented framework with stronger emphasis on complex runtime integration. The broader trend across these systems is a movement from generic RL infrastructure toward increasingly agent-native training abstractions. Agent-R1 is positioned in this latter direction. Its main emphasis is the training abstraction itself: step-level MDP, structured step-native traces, flexible context management, and interfaces that let multiple optimization algorithms share the same multi-turn substrate [2]. In this sense, Agent-R1 is less about one specific optimizer and more about making agentic RL trainable through a coherent framework design. 

## 8 Conclusion

In this paper, we first revisited how LLM training extends into RL training, and why the transition from single-turn post-training to agentic RL brings new challenges in environment adaptation, rollout structure, trajectory representation, and context construction. We then introduced Agent-R1 as a unified and modular framework for multi-turn RL training of LLM agents, with emphasis on step-level MDP abstraction, structured step-level trajectories, flexible context management, unified interfaces across workflows and infrastructure, and compatibility with multiple optimization algorithms such as PPO and GRPO. We further showed, through a controlled GSM8K case study and the broader framework design throughout the paper, how these ideas make it possible to support diferent algorithmic instantiations on top of a common multi-turn RL substrate while keeping the training process analyzable, extensible, and reusable. As LLM agents continue to grow in capability and complexity, we hope Agent-R1 can provide the community with a principled, modular, and easy-to-use foundation for building, studying, and extending future agentic RL systems. 

## References



[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 





[2] AgentR1. Agent-r1 github repository. https://github.com/AgentR1/Agent-R1, 2026. 





[3] AgentR1. Claw-r1 github repository. https://github.com/AgentR1/Claw-R1, 2026. 





[4] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 





[5] Mingyue Cheng, Daoyu Wang, Shuo Yu, Qingchuan Li, Jie Ouyang, Yucong Luo, Yiju Zhang, Qi Liu, and Enhong Chen. A comprehensive survey of the llm-based agent: The contextual cognition perspective. 2026. 





[6] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. 





[7] Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-group policy optimization for LLM agent training. arXiv preprint arXiv:2505.10978, 2025. 





[8] Wei Fu, Jiaxuan Gao, Xujie Shen, Chen Zhu, Zhiyu Mei, Chuyi He, Shusheng Xu, Guo Wei, Jun Mei, Jiashu Wang, et al. AReaL: A large-scale asynchronous reinforcement learning system for language reasoning. arXiv preprint arXiv:2505.24298, 2025. 





[9] Yuxiang Ji, Ziyu Ma, Yong Wang, Guanhua Chen, Xiangxiang Chu, and Liaoni Wu. Tree search for LLM agent reinforcement learning. arXiv preprint arXiv:2509.21240, 2025. 





[10] Dongfu Jiang, Yi Lu, Zhuofeng Li, Zhiheng Lyu, Ping Nie, Haozhe Wang, Alex Su, Hui Chen, Kai Zou, Chao Du, et al. VerlTool: Towards holistic agentic reinforcement learning with tool use. arXiv preprint arXiv:2509.01055, 2025. 





[11] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-R1: Training LLMs to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025. 





[12] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Eficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023. 





[13] Junbo Li, Peng Zhou, Rui Meng, Meet P Vadera, Lihong Li, and Yang Li. Turn-PPO: Turn-level advantage estimation with PPO for improved multi-turn RL in agentic LLMs. In Findings of the Association for Computational Linguistics: EACL 2026, pages 6227–6243, 2026. 





[14] Qingchuan Li, Jiatong Li, Zirui Liu, Mingyue Cheng, Yitong Zhou, Yuting Zeng, Qi Liu, and Tongxuan Liu. Are llms stable formal logic translators in logical reasoning across linguistically diversified texts? In Proceedings of the ACM Web Conference 2026, pages 3633–3644, 2026. 





[15] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. Agentbench: Evaluating llms as agents. arXiv preprint arXiv:2308.03688, 2023. 





[16] Xufang Luo, Yuge Zhang, Zhiyuan He, Zilong Wang, Siyun Zhao, Dongsheng Li, Luna K. Qiu, and Yuqing Yang. Agent lightning: Train ANY AI agents with reinforcement learning. arXiv preprint arXiv:2508.03680, 2025. 





[17] Microsoft. Deepspeed github repository. https://github.com/microsoft/DeepSpeed, 2026. 





[18] MiniMax. Forge: Scalable agent rl framework and algorithm. https://www.minimax.io/news/ forge-scalable-agent-rl-framework-and-algorithm, 2026. 





[19] Long Ouyang, Jefrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, 2022. 





[20] Tingyue Pan, Jie Ouyang, Mingyue Cheng, Qingchuan Li, Zirui Liu, Daoyu Wang, Mingfan Pan, Shuo Yu, and Qi Liu. Paperscout: An autonomous agent for academic paper search with process-aware sequence-level policy optimization. arXiv preprint arXiv:2601.10029, 2026. 





[21] rLLM. rllm documentation. https://docs.rllm-project.com/, 2026. 





[22] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in neural information processing systems, 36:68539–68551, 2023. 





[23] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 





[24] SGLang Team. Sglang: Eficient structured generation for large language models. https://github.com/ sgl-project/sglang, 2024. 





[25] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 





[26] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and eficient RLHF framework. In Proceedings of the Twentieth European Conference on Computer Systems, 2025. doi: 10.1145/3689031.3696075. 





[27] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019. 





[28] Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. arXiv preprint arXiv:2010.03768, 2020. 





[29] The slime Team. slime: An SGLang-native post-training framework for RL scaling. https://lmsys.org/blog/ 2025-07-09-slime/, 2025. 





[30] Daoyu Wang, Qingchuan Li, Mingyue Cheng, Jie Ouyang, Shuo Yu, Qi Liu, and Enhong Chen. Steppo: Step-aligned policy optimization for agentic reinforcement learning. arXiv preprint arXiv:2604.18401, 2026. 





[31] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023. 





[32] Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Xing Jin, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, et al. RAGEN: Understanding self-evolution in LLM agents via multi-turn reinforcement learning. arXiv preprint arXiv:2504.20073, 2025. 





[33] Zhiheng Xi, Chenyang Liao, Guanyu Li, Zhihao Zhang, Wenxiang Chen, Binghai Wang, Senjie Jin, Yuhao Zhou, Jian Guan, Wei Wu, et al. Agentprm: Process reward models for LLM agents via step-wise promise and progress. In Proceedings of the ACM Web Conference 2026, pages 4184–4195, 2026. 





[34] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 





[35] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2369–2380, 2018. 





[36] Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents. Advances in Neural Information Processing Systems, 35:20744–20757, 2022. 





[37] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022. 





[38] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source LLM reinforcement learning system at scale. arXiv preprin arXiv:2503.14476, 2025. 





[39] Shuo Yu, Mingyue Cheng, Qi Liu, Daoyu Wang, Jiqian Yang, Jie Ouyang, Yucong Luo, Chenyi Lei, and Enhong Chen. Multi-source knowledge pruning for retrieval-augmented generation: A benchmark and empirical study. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management, pages 3931–3941, 2025. 





[40] Guibin Zhang, Hejia Geng, Xiaohang Yu, Zhenfei Yin, Zaibin Zhang, Zelin Tan, Heng Zhou, Zhongzhi Li, Xiangyuan Xue, Yijiang Li, et al. The landscape of agentic reinforcement learning for LLMs: A survey. arXiv preprint arXiv:2509.02547, 2025. 





[41] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023. 





[42] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025. 





[43] Yifei Zhou, Song Jiang, Yuandong Tian, Jason Weston, Sergey Levine, Sainbayar Sukhbaatar, and Xian Li. SWEET-RL: Training multi-turn LLM agents on collaborative reasoning tasks. arXiv preprint arXiv:2503.15478, 2025. 

